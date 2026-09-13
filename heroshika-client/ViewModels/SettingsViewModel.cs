using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using heroshika_client.Models;
using heroshika_client.Services;
using Microsoft.Extensions.Logging;

namespace heroshika_client.ViewModels;

public partial class SettingsViewModel : BaseViewModel
{
    private readonly ISettingsService _settingsService;
    private readonly IHeroshikoApiService _apiService;
    private readonly ILogger<SettingsViewModel> _logger;

    [ObservableProperty]
    public partial string ServerUrl { get; set; } = string.Empty;

    [ObservableProperty]
    public partial bool IsTesting { get; set; }

    [ObservableProperty]
    public partial bool IsOnline { get; set; }

    [ObservableProperty]
    public partial string TestStatus { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string GpuInfo { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string VramInfo { get; set; } = string.Empty;

    public SettingsViewModel(
        ISettingsService settingsService,
        IHeroshikoApiService apiService,
        ILogger<SettingsViewModel> logger)
    {
        _settingsService = settingsService;
        _apiService = apiService;
        _logger = logger;
        Title = "Настройки студии";
    }

    [RelayCommand]
    public void Initialize()
    {
        ServerUrl = _settingsService.ServerUrl;
    }

    [RelayCommand]
    public void SelectCloud()
    {
        ServerUrl = SettingsService.CloudTunnelUrl;
        _settingsService.ServerUrl = ServerUrl;
    }

    [RelayCommand]
    public void SelectLan()
    {
        ServerUrl = SettingsService.LocalLanUrl;
        _settingsService.ServerUrl = ServerUrl;
    }

    [RelayCommand]
    public void SelectUsb()
    {
        ServerUrl = SettingsService.LocalUsbUrl;
        _settingsService.ServerUrl = ServerUrl;
    }

    [RelayCommand]
    public async Task AutoDetectAsync()
    {
        IsTesting = true;
        TestStatus = "Автоматический поиск сервера...";
        var found = await _apiService.AutoDiscoverServerAsync();
        if (!string.IsNullOrEmpty(found))
        {
            ServerUrl = found;
            await TestConnectionAsync();
        }
        else
        {
            IsTesting = false;
            TestStatus = "❌ Сервер не отвечает. Запустите run_server.py на ПК.";
        }
    }

    [RelayCommand]
    public async Task TestConnectionAsync()
    {
        try
        {
            IsTesting = true;
            TestStatus = "Проверка соединения...";
            GpuInfo = string.Empty;
            VramInfo = string.Empty;

            _settingsService.ServerUrl = ServerUrl;

            var health = await _apiService.GetHealthAsync();
            if (health != null && health.IsOnline)
            {
                IsOnline = true;
                TestStatus = $"✅ Сервер онлайн (версия {health.Version})";
                GpuInfo = $"Устройство: {health.DeviceName ?? "CUDA"}";
                VramInfo = $"Видеопамять: {health.VramSummary}";
            }
            else
            {
                IsOnline = false;
                TestStatus = "❌ Сервер не отвечает или вернул ошибку";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Test connection error");
            IsOnline = false;
            TestStatus = $"❌ Ошибка соединения: {ex.Message}";
        }
        finally
        {
            IsTesting = false;
        }
    }

    [RelayCommand]
    public void ResetDefault()
    {
        _settingsService.ResetToDefault();
        ServerUrl = _settingsService.ServerUrl;
        TestStatus = "Сброшено к значению по умолчанию.";
        IsOnline = false;
        GpuInfo = string.Empty;
        VramInfo = string.Empty;
    }

    [RelayCommand]
    public void SaveSettings()
    {
        _settingsService.ServerUrl = ServerUrl;
        TestStatus = "Настройки успешно сохранены.";
    }
}
