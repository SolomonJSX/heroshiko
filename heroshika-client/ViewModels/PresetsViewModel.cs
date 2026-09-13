using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using heroshika_client.Models;
using heroshika_client.Services;
using Microsoft.Extensions.Logging;

namespace heroshika_client.ViewModels;

public partial class PresetsViewModel : BaseViewModel
{
    private readonly IHeroshikoApiService _apiService;
    private readonly IWorkflowContext _workflowContext;
    private readonly ILogger<PresetsViewModel> _logger;

    public ObservableCollection<PresetResponse> Presets { get; } = new();

    [ObservableProperty]
    public partial PresetResponse? SelectedPreset { get; set; }

    [ObservableProperty]
    public partial bool IsServerOnline { get; set; }

    [ObservableProperty]
    public partial string ServerStatusText { get; set; } = "Поиск сервера...";

    [ObservableProperty]
    public partial string ServerDetails { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string ErrorMessage { get; set; } = string.Empty;

    public PresetsViewModel(
        IHeroshikoApiService apiService,
        IWorkflowContext workflowContext,
        ILogger<PresetsViewModel> logger)
    {
        _apiService = apiService;
        _workflowContext = workflowContext;
        _logger = logger;
        Title = "Каталог студии";

        // Сразу загружаем встроенные стили, чтобы экран никогда не был пустым
        PopulateDefaultPresets();
    }

    private void PopulateDefaultPresets()
    {
        if (Presets.Count > 0) return;

        Presets.Add(new PresetResponse
        {
            Id = "dubai_luxury",
            Title = "Дубай: Роскошь и Золотой Час",
            Prompt = "Роскошный кинематографичный портрет в Дубай Марина во время заката, стильный наряд, студийный свет, естественная текстура кожи",
            Tags = new List<string> { "lifestyle", "luxury" }
        });
        Presets.Add(new PresetResponse
        {
            Id = "paris_cafe",
            Title = "Париж: Уютное Кафе Монмартр",
            Prompt = "Атмосферная фотосессия в парижском уличном кафе, мягкий утренний свет, непринужденный парижский шик",
            Tags = new List<string> { "lifestyle", "portrait" }
        });
        Presets.Add(new PresetResponse
        {
            Id = "kpop_idol",
            Title = "K-Pop Idol: Соло Сцена",
            Prompt = "Гламурный портрет K-Pop айдола на сцене, динамичный неоновый свет, сценический образ, харизматичный взгляд",
            Tags = new List<string> { "kpop", "music" }
        });
        Presets.Add(new PresetResponse
        {
            Id = "kpop_group",
            Title = "K-Pop Группа: Студийный Постер",
            Prompt = "Стильная фотосессия для музыкального альбома K-Pop группы, дизайнерские костюмы, идеальное студийное освещение",
            Tags = new List<string> { "group", "kpop" }
        });
    }

    [RelayCommand]
    public async Task InitializeAsync()
    {
        await CheckServerHealthAsync();
        await LoadPresetsAsync();
    }

    [RelayCommand]
    public async Task CheckServerHealthAsync()
    {
        try
        {
            var health = await _apiService.GetHealthAsync();
            MainThread.BeginInvokeOnMainThread(() =>
            {
                if (health != null && health.IsOnline)
                {
                    IsServerOnline = true;
                    ServerStatusText = "Сервер онлайн";
                    ServerDetails = $"{health.DeviceName ?? "CUDA"} ({health.VramSummary})";
                }
                else
                {
                    IsServerOnline = false;
                    ServerStatusText = "Сервер оффлайн";
                    ServerDetails = "Запустите run_server.py";
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check error");
            MainThread.BeginInvokeOnMainThread(() =>
            {
                IsServerOnline = false;
                ServerStatusText = "Поиск сервера...";
                ServerDetails = ex.Message;
            });
        }
    }

    [RelayCommand]
    public async Task LoadPresetsAsync()
    {
        if (IsBusy) return;

        try
        {
            MainThread.BeginInvokeOnMainThread(() => IsBusy = true);
            ErrorMessage = string.Empty;

            var list = await _apiService.GetPresetsAsync();
            MainThread.BeginInvokeOnMainThread(() =>
            {
                if (list.Count > 0)
                {
                    Presets.Clear();
                    foreach (var preset in list)
                    {
                        Presets.Add(preset);
                    }
                }
                else if (Presets.Count == 0)
                {
                    PopulateDefaultPresets();
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to fetch presets online, keeping built-in presets");
            MainThread.BeginInvokeOnMainThread(() =>
            {
                if (Presets.Count == 0)
                {
                    PopulateDefaultPresets();
                }
            });
        }
        finally
        {
            MainThread.BeginInvokeOnMainThread(() => IsBusy = false);
        }
    }

    [RelayCommand]
    public async Task SelectPresetAsync(PresetResponse preset)
    {
        if (preset == null) return;
        SelectedPreset = preset;
        _workflowContext.SelectedPreset = preset;
        _workflowContext.GenerationParams.PresetId = preset.Id;

        await Shell.Current.GoToAsync("FaceAnalysisPage");
    }

    [RelayCommand]
    public async Task OpenSettingsAsync()
    {
        await Shell.Current.GoToAsync("SettingsPage");
    }
}
