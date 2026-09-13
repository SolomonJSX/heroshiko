using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using heroshika_client.Models;
using heroshika_client.Services;
using Microsoft.Extensions.Logging;

namespace heroshika_client.ViewModels;

public partial class GenerationProgressViewModel : BaseViewModel
{
    private readonly IHeroshikoApiService _apiService;
    private readonly IWorkflowContext _workflowContext;
    private readonly ILogger<GenerationProgressViewModel> _logger;
    private CancellationTokenSource? _pollCts;

    [ObservableProperty]
    public partial string TaskId { get; set; } = string.Empty;

    [ObservableProperty]
    public partial int Progress { get; set; }

    [ObservableProperty]
    public partial double ProgressDouble { get; set; }

    [ObservableProperty]
    public partial string StatusText { get; set; } = "Инициализация...";

    [ObservableProperty]
    public partial string StatusDetail { get; set; } = "Отправка селфи и параметров на сервер...";

    [ObservableProperty]
    public partial bool IsGenerating { get; set; } = true;

    [ObservableProperty]
    public partial bool IsCompleted { get; set; }

    [ObservableProperty]
    public partial bool IsFailed { get; set; }

    [ObservableProperty]
    public partial string ErrorMessage { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string ResultImageUrl { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string FaceImageUrl { get; set; } = string.Empty;

    [ObservableProperty]
    public partial float ElapsedSeconds { get; set; }

    public GenerationProgressViewModel(
        IHeroshikoApiService apiService,
        IWorkflowContext workflowContext,
        ILogger<GenerationProgressViewModel> logger)
    {
        _apiService = apiService;
        _workflowContext = workflowContext;
        _logger = logger;
        Title = "Генерация шедевра";
    }

    [RelayCommand]
    public async Task StartProcessAsync()
    {
        if (_workflowContext.SelectedPhotoBytes == null)
        {
            StatusText = "Ошибка";
            StatusDetail = "Селфи не выбрано!";
            IsFailed = true;
            IsGenerating = false;
            return;
        }

        try
        {
            IsGenerating = true;
            IsCompleted = false;
            IsFailed = false;
            Progress = 5;
            ProgressDouble = 0.05;
            StatusText = "Запуск задачи...";
            StatusDetail = "Отправка данных на GPU сервер...";

            using var stream = new MemoryStream(_workflowContext.SelectedPhotoBytes);
            var task = await _apiService.StartGenerationAsync(
                stream,
                "selfie.jpg",
                _workflowContext.GenerationParams
            );

            if (task == null)
            {
                throw new Exception("Не удалось создать задачу генерации.");
            }

            TaskId = task.TaskId;
            _workflowContext.CurrentTask = task;

            _pollCts = new CancellationTokenSource();
            await PollTaskStatusAsync(_pollCts.Token);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute generation process");
            ErrorMessage = ex.Message;
            StatusText = "Ошибка генерации";
            StatusDetail = ex.Message;
            IsFailed = true;
            IsGenerating = false;
        }
    }

    private async Task PollTaskStatusAsync(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(1500, token);

                var task = await _apiService.GetTaskStatusAsync(TaskId);
                if (task == null) continue;

                Progress = task.Progress;
                ProgressDouble = task.Progress / 100.0;

                if (task.Status.Equals("queued", StringComparison.OrdinalIgnoreCase))
                {
                    StatusText = "В очереди...";
                    StatusDetail = "Ожидание освобождения GPU...";
                }
                else if (task.Status.Equals("processing", StringComparison.OrdinalIgnoreCase))
                {
                    StatusText = "Идет генерация...";
                    StatusDetail = $"Нейросеть рисует фотосессию ({task.Progress}%)...";
                }
                else if (task.IsCompleted)
                {
                    Progress = 100;
                    ProgressDouble = 1.0;
                    StatusText = "Готово!";
                    StatusDetail = $"Фотосессия сгенерирована за {task.ElapsedSeconds:F1} сек.";
                    ResultImageUrl = task.FullResultUrl;
                    FaceImageUrl = task.FullFaceUrl;
                    ElapsedSeconds = task.ElapsedSeconds ?? 0f;

                    IsGenerating = false;
                    IsCompleted = true;
                    break;
                }
                else if (task.IsFailed)
                {
                    StatusText = "Ошибка создания";
                    StatusDetail = task.ErrorMessage ?? "Неизвестная ошибка";
                    ErrorMessage = task.ErrorMessage ?? "Ошибка";
                    IsGenerating = false;
                    IsFailed = true;
                    break;
                }
            }
            catch (TaskCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error polling status");
            }
        }
    }

    [RelayCommand]
    public async Task SaveImageAsync()
    {
        if (string.IsNullOrEmpty(ResultImageUrl)) return;

        try
        {
            using var client = new HttpClient();
            var bytes = await client.GetByteArrayAsync(ResultImageUrl);

            string targetDir;
            if (DeviceInfo.Platform == DevicePlatform.Android)
            {
                // На Android сохраняем в папку загрузок / кеш или AppData
                targetDir = FileSystem.AppDataDirectory;
            }
            else
            {
                targetDir = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
                if (string.IsNullOrEmpty(targetDir))
                    targetDir = FileSystem.CacheDirectory;
            }

            var filename = $"heroshiko_{TaskId}_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
            var filePath = Path.Combine(targetDir, filename);

            await File.WriteAllBytesAsync(filePath, bytes);
            StatusDetail = $"Сохранено: {filename}";

            if (Shell.Current != null)
            {
                await Shell.Current.DisplayAlertAsync("Успешно", $"Фото сохранено:\n{filePath}", "OK");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save image");
            if (Shell.Current != null)
            {
                await Shell.Current.DisplayAlertAsync("Ошибка", $"Не удалось сохранить фото: {ex.Message}", "OK");
            }
        }
    }

    [RelayCommand]
    public async Task ShareImageAsync()
    {
        if (string.IsNullOrEmpty(ResultImageUrl)) return;

        try
        {
            using var client = new HttpClient();
            var bytes = await client.GetByteArrayAsync(ResultImageUrl);

            var tempPath = Path.Combine(FileSystem.CacheDirectory, $"heroshiko_{TaskId}.jpg");
            await File.WriteAllBytesAsync(tempPath, bytes);

            await Share.Default.RequestAsync(new ShareFileRequest
            {
                Title = "Поделиться фотосессией Heroshiko",
                File = new ShareFile(tempPath)
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to share image");
            if (Shell.Current != null)
            {
                await Shell.Current.DisplayAlertAsync("Ошибка", $"Не удалось поделиться фото: {ex.Message}", "OK");
            }
        }
    }

    [RelayCommand]
    public async Task NewPhotoshootAsync()
    {
        _workflowContext.Reset();
        await Shell.Current.GoToAsync("//PresetsPage");
    }

    public void CancelPolling()
    {
        _pollCts?.Cancel();
        _pollCts = null;
    }
}
