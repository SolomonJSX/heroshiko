import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from heroshiko.api.routers import analyze, generate, health, images, presets
from heroshiko.services.generation_service import HeroshikoService
from heroshiko.services.task_manager import TaskManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("heroshiko.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл FastAPI приложения (инициализация и очистка ресурсов)."""
    logger.info("[START] Запуск Heroshiko REST API сервиса...")

    # Создание директорий хранилища
    Path("storage/outputs").mkdir(parents=True, exist_ok=True)
    Path("storage/uploads").mkdir(parents=True, exist_ok=True)

    # Инициализация сервиса и фонового воркера задач
    HeroshikoService.get_instance()
    task_mgr = TaskManager.get_instance()
    task_mgr.start_worker()

    logger.info("[READY] Сервис готов к приему запросов!")
    yield

    logger.info("[STOP] Остановка сервиса...")
    task_mgr.stop_worker()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Heroshiko AI Photo Studio API",
        description=(
            "Высокопроизводительный REST API бэкенд для фотосессий нового поколения.\n\n"
            "Ключевые возможности:\n"
            "- 🎭 Каталог пресетов (Дубай, Париж, K-Pop Соло, K-Pop Группа);\n"
            "- 👤 Мгновенная биометрическая пред-диагностика селфи (пол, возраст, прическа, телосложение);\n"
            "- ⚡ Асинхронная очередь генерации с трекингом прогресса;\n"
            "- 💎 100% перенос биометрии лица и реалистичная текстура кожи (INSwapper + CodeFormer + Film Grain);\n"
            "- 🛡️ Потокобезопасная изоляция GPU для предотвращения CUDA OOM."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware для работы с мобильными приложениями (Flutter, React Native) и веб-клиентами
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Подключение роутов API v1
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(presets.router, prefix=api_prefix)
    app.include_router(analyze.router, prefix=api_prefix)
    app.include_router(generate.router, prefix=api_prefix)
    app.include_router(images.router, prefix=api_prefix)

    # Алиас для корневого healthcheck (стандарт для Docker/K8s)
    @app.get("/health", include_in_schema=False)
    async def root_health():
        return await health.get_health()

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_app()
