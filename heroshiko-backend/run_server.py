import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Добавляем папку src в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
import uvicorn

# Загружаем переменные окружения из .env если файл существует
load_dotenv(PROJECT_ROOT / ".env")


def setup_ngrok_tunnel(port: int, token: str | None = None) -> str | None:
    """Инициализирует и запускает публичный HTTPS туннель ngrok."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("❌ Ошибка: пакет 'pyngrok' не установлен. Установите командой: uv add pyngrok")
        return None

    auth_token = token or os.environ.get("NGROK_AUTHTOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token.strip())
    else:
        print("\n⚠️ ВНИМАНИЕ: NGROK_AUTHTOKEN не указан!")
        print("💡 Для работы ngrok нужен бесплатный токен:")
        print("   1. Зарегистрируйтесь на https://dashboard.ngrok.com")
        print("   2. Скопируйте токен на https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. Добавьте его в .env (NGROK_AUTHTOKEN=ваш_токен) или передайте флаг --ngrok-token <токен>\n")

    domain = os.environ.get("NGROK_DOMAIN", "carnage-pregame-striking.ngrok-free.dev")
    try:
        tunnel = ngrok.connect(port, "http", domain=domain)
        return tunnel.public_url
    except Exception as e:
        print(f"⚠️ Не удалось подключиться к статическому домену {domain}: {e}")
        try:
            tunnel = ngrok.connect(port, "http")
            return tunnel.public_url
        except Exception as e2:
            print(f"❌ Ошибка подключения ngrok туннеля: {e2}")
            return None


def main():
    default_host = os.environ.get("HEROSHIKO_HOST", "0.0.0.0")
    default_port = int(os.environ.get("HEROSHIKO_PORT", "8000"))

    parser = argparse.ArgumentParser(description="Запуск Heroshiko REST API сервера")
    parser.add_argument("--host", type=str, default=default_host, help=f"IP адрес (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Порт сервера (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Авто-перезагрузка при изменении кода")
    parser.add_argument("--tunnel", action="store_true", help="Запустить публичный HTTPS туннель через ngrok")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok authtoken (или задайте NGROK_AUTHTOKEN в .env)")

    args = parser.parse_args()

    public_url = None
    if args.tunnel or args.ngrok_token:
        public_url = setup_ngrok_tunnel(args.port, args.ngrok_token)

    print("\n" + "=" * 65)
    print(f"🚀 Heroshiko API локально:  http://127.0.0.1:{args.port}")
    print(f"📖 Swagger UI (локальный): http://127.0.0.1:{args.port}/docs")
    if public_url:
        print("-" * 65)
        print(f"🌍 ПУБЛИЧНЫЙ URL (ngrok):  {public_url}")
        print(f"📖 Swagger UI (публичный): {public_url}/docs")
        print(f"📲 Клиенты могут отправлять запросы прямо на {public_url}/api/v1/...")
    print("=" * 65 + "\n")

    try:
        uvicorn.run(
            "heroshiko.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    finally:
        if public_url:
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
