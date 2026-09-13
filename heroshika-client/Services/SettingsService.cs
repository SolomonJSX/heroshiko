namespace heroshika_client.Services;

public interface ISettingsService
{
    string ServerUrl { get; set; }
    IReadOnlyList<string> CandidateUrls { get; }
    void ResetToDefault();
}

public class SettingsService : ISettingsService
{
    private const string ServerUrlKey = "heroshiko_server_url";

    // Постоянный публичный туннель пользователя (доступен из любой точки мира через 4G/5G/Wi-Fi)
    public const string CloudTunnelUrl = "https://carnage-pregame-striking.ngrok-free.dev";

    // Локальный адрес компьютера в домашней Wi-Fi сети
    public const string LocalLanUrl = "http://192.168.0.103:8000";

    // Адрес по USB кабелю (adb reverse)
    public const string LocalUsbUrl = "http://localhost:8000";

    // Адрес для Android эмулятора
    public const string EmulatorUrl = "http://10.0.2.2:8000";

    public static string DefaultUrl => CloudTunnelUrl;

    public IReadOnlyList<string> CandidateUrls => new[]
    {
        CloudTunnelUrl,
        LocalLanUrl,
        LocalUsbUrl,
        EmulatorUrl
    };

    public string ServerUrl
    {
        get => Preferences.Default.Get(ServerUrlKey, DefaultUrl);
        set
        {
            var trimmed = value?.TrimEnd('/') ?? DefaultUrl;
            Preferences.Default.Set(ServerUrlKey, trimmed);
        }
    }

    public void ResetToDefault()
    {
        Preferences.Default.Remove(ServerUrlKey);
    }
}
