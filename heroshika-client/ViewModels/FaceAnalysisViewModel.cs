using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using heroshika_client.Models;
using heroshika_client.Services;
using Microsoft.Extensions.Logging;

namespace heroshika_client.ViewModels;

public partial class FaceAnalysisViewModel : BaseViewModel
{
    private readonly IHeroshikoApiService _apiService;
    private readonly IWorkflowContext _workflowContext;
    private readonly ILogger<FaceAnalysisViewModel> _logger;

    [ObservableProperty]
    public partial string SelectedPresetTitle { get; set; } = string.Empty;

    [ObservableProperty]
    public partial ImageSource? PreviewImage { get; set; }

    [ObservableProperty]
    public partial bool HasPhoto { get; set; }

    [ObservableProperty]
    public partial FaceAnalysisResponse? AnalysisResult { get; set; }

    [ObservableProperty]
    public partial bool IsAnalyzing { get; set; }

    [ObservableProperty]
    public partial string StatusMessage { get; set; } = "Загрузите селфи для мгновенной AI-диагностики лица";

    [ObservableProperty]
    public partial bool IsReadyToTune { get; set; }

    public FaceAnalysisViewModel(
        IHeroshikoApiService apiService,
        IWorkflowContext workflowContext,
        ILogger<FaceAnalysisViewModel> logger)
    {
        _apiService = apiService;
        _workflowContext = workflowContext;
        _logger = logger;
        Title = "Анализ селфи";
    }

    [RelayCommand]
    public void Initialize()
    {
        SelectedPresetTitle = _workflowContext.SelectedPreset?.Title ?? "Стиль не выбран";
        if (_workflowContext.SelectedPhotoBytes != null)
        {
            PreviewImage = ImageSource.FromStream(() => new MemoryStream(_workflowContext.SelectedPhotoBytes));
            HasPhoto = true;
            AnalysisResult = _workflowContext.AnalysisResult;
            IsReadyToTune = AnalysisResult?.FaceDetected ?? false;
        }
    }

    [RelayCommand]
    public async Task PickPhotoAsync()
    {
        try
        {
            var options = new PickOptions
            {
                PickerTitle = "Выберите селфи",
                FileTypes = FilePickerFileType.Images
            };

            var result = await FilePicker.Default.PickAsync(options);
            if (result == null) return;

            await ProcessPickedFileAsync(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to pick image");
            StatusMessage = $"Ошибка выбора фото: {ex.Message}";
        }
    }

    [RelayCommand]
    public async Task TakePhotoAsync()
    {
        try
        {
            if (MediaPicker.Default.IsCaptureSupported)
            {
                var photo = await MediaPicker.Default.CapturePhotoAsync();
                if (photo == null) return;

                await ProcessPickedFileAsync(photo);
            }
            else
            {
                StatusMessage = "Камера недоступна на данном устройстве.";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to capture photo");
            StatusMessage = $"Ошибка камеры: {ex.Message}";
        }
    }

    private async Task ProcessPickedFileAsync(FileResult file)
    {
        try
        {
            IsBusy = true;
            IsAnalyzing = true;
            StatusMessage = "Обработка изображения...";

            using var stream = await file.OpenReadAsync();
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);
            var bytes = ms.ToArray();

            _workflowContext.SelectedPhotoBytes = bytes;
            _workflowContext.SelectedPhotoPath = file.FullPath;

            PreviewImage = ImageSource.FromStream(() => new MemoryStream(bytes));
            HasPhoto = true;

            StatusMessage = "Анализ биометрии (InsightFace & HairNet)...";
            using var uploadStream = new MemoryStream(bytes);
            var analysis = await _apiService.AnalyzeFaceAsync(uploadStream, file.FileName);

            AnalysisResult = analysis;
            _workflowContext.AnalysisResult = analysis;

            if (analysis != null && analysis.FaceDetected)
            {
                StatusMessage = "Лицо успешно распознано!";
                IsReadyToTune = true;

                if (!string.IsNullOrEmpty(analysis.BuildType))
                {
                    _workflowContext.GenerationParams.BodyBuild = analysis.BuildType;
                }
            }
            else
            {
                StatusMessage = "⚠️ Лицо не найдено! Пожалуйста, выберите четкое фото лица анфас.";
                IsReadyToTune = false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to analyze image");
            StatusMessage = $"Ошибка анализа: {ex.Message}";
            IsReadyToTune = false;
        }
        finally
        {
            IsAnalyzing = false;
            IsBusy = false;
        }
    }

    [RelayCommand]
    public async Task ProceedToTuningAsync()
    {
        if (!IsReadyToTune) return;
        await Shell.Current.GoToAsync("GenerationTuningPage");
    }

    [RelayCommand]
    public async Task GoBackAsync()
    {
        await Shell.Current.GoToAsync("..");
    }
}
