using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using heroshika_client.Models;
using Microsoft.Extensions.Logging;

namespace heroshika_client.Services;

public interface IHeroshikoApiService
{
    Task<ServerHealthResponse?> GetHealthAsync();
    Task<string?> AutoDiscoverServerAsync();
    Task<List<PresetResponse>> GetPresetsAsync();
    Task<FaceAnalysisResponse?> AnalyzeFaceAsync(Stream imageStream, string fileName);
    Task<TaskResponse?> StartGenerationAsync(Stream imageStream, string fileName, GenerationParams parameters);
    Task<TaskResponse?> GetTaskStatusAsync(string taskId);
    string GetFullUrl(string? pathOrUrl);
}

public class HeroshikoApiService : IHeroshikoApiService
{
    private readonly HttpClient _httpClient;
    private readonly ISettingsService _settingsService;
    private readonly ILogger<HeroshikoApiService> _logger;

    public HeroshikoApiService(
        HttpClient httpClient,
        ISettingsService settingsService,
        ILogger<HeroshikoApiService> logger)
    {
        _httpClient = httpClient;
        _settingsService = settingsService;
        _logger = logger;
        _httpClient.Timeout = TimeSpan.FromSeconds(120);

        // Пропускаем приветственную HTML страницу Ngrok при вызовах REST API
        _httpClient.DefaultRequestHeaders.TryAddWithoutValidation("ngrok-skip-browser-warning", "true");
        _httpClient.DefaultRequestHeaders.TryAddWithoutValidation("User-Agent", "HeroshikoClient/1.0");
    }

    public string GetFullUrl(string? pathOrUrl)
    {
        if (string.IsNullOrWhiteSpace(pathOrUrl))
            return string.Empty;

        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            return pathOrUrl;
        }

        var baseUrl = _settingsService.ServerUrl.TrimEnd('/');
        var relative = pathOrUrl.TrimStart('/');
        return $"{baseUrl}/{relative}";
    }

    public async Task<string?> AutoDiscoverServerAsync()
    {
        var candidates = new List<string>();

        // Если устройство в мобильной сети (Cellular) — облачный туннель должен идти первым
        var profiles = Connectivity.Current.ConnectionProfiles.ToList();
        bool isCellular = profiles.Contains(ConnectionProfile.Cellular);

        if (isCellular)
        {
            candidates.Add(SettingsService.CloudTunnelUrl);
        }

        // 1. Текущий сохраненный URL
        if (!string.IsNullOrEmpty(_settingsService.ServerUrl))
            candidates.Add(_settingsService.ServerUrl);

        // 2. Все системные кандидаты (Облачный туннель, Домашний Wi-Fi, USB, Эмулятор)
        candidates.AddRange(_settingsService.CandidateUrls);

        var distinct = candidates.Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        // Проверяем всех кандидатов параллельно (первый ответивший побеждает)
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(8));
        var checkTasks = distinct.Select(async candidate =>
        {
            try
            {
                // Для облачного туннеля через мобильные 4G данные даем 4.5 сек на TLS-рукопожатие
                var timeoutMs = candidate.StartsWith("https", StringComparison.OrdinalIgnoreCase) ? 4500 : 1500;
                using var singleCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(timeoutMs));
                using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(singleCts.Token, cts.Token);
                var testUrl = $"{candidate.TrimEnd('/')}/api/v1/health";
                var response = await _httpClient.GetAsync(testUrl, linkedCts.Token);
                if (response.IsSuccessStatusCode)
                {
                    return candidate;
                }
            }
            catch
            {
            }
            return null;
        }).ToList();

        while (checkTasks.Count > 0)
        {
            var finished = await Task.WhenAny(checkTasks);
            checkTasks.Remove(finished);
            var result = await finished;
            if (!string.IsNullOrEmpty(result))
            {
                _logger.LogInformation("Auto-discovered working server: {Url}", result);
                _settingsService.ServerUrl = result;
                cts.Cancel(); // Останавливаем остальные проверки
                return result;
            }
        }

        return null;
    }

    public async Task<ServerHealthResponse?> GetHealthAsync()
    {
        try
        {
            // Умная смена сети: если перешли на мобильные данные (Cellular), а URL локальный (192.168 / localhost) — сразу берем Облако
            var profiles = Connectivity.Current.ConnectionProfiles.ToList();
            bool isCellularOnly = profiles.Contains(ConnectionProfile.Cellular) && !profiles.Contains(ConnectionProfile.WiFi);
            if (isCellularOnly && (_settingsService.ServerUrl.Contains("192.168.") || _settingsService.ServerUrl.Contains("localhost") || _settingsService.ServerUrl.Contains("10.0.2.2")))
            {
                _logger.LogInformation("Cellular network detected with local URL, auto-switching to Cloud Tunnel");
                _settingsService.ServerUrl = SettingsService.CloudTunnelUrl;
            }

            var url = GetFullUrl("/api/v1/health");
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var response = await _httpClient.GetAsync(url, cts.Token);
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<ServerHealthResponse>(json);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Direct health check failed, initiating auto-discovery...");
        }

        // Если текущий URL не ответил — пробуем умный поиск сервера среди кандидатов
        var discoveredUrl = await AutoDiscoverServerAsync();
        if (!string.IsNullOrEmpty(discoveredUrl))
        {
            try
            {
                var url = $"{discoveredUrl.TrimEnd('/')}/api/v1/health";
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                var response = await _httpClient.GetAsync(url, cts.Token);
                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<ServerHealthResponse>(json);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to query health after discovery");
            }
        }

        return null;
    }

    public async Task<List<PresetResponse>> GetPresetsAsync()
    {
        try
        {
            var url = GetFullUrl("/api/v1/presets");
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<PresetListResponse>(json);
            if (result?.Presets == null)
                return new List<PresetResponse>();

            foreach (var p in result.Presets)
            {
                if (!string.IsNullOrEmpty(p.CoverUrl))
                {
                    p.FullCoverUrl = GetFullUrl(p.CoverUrl);
                }
            }

            return result.Presets;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to fetch presets directly, trying auto-discovery...");
            var discovered = await AutoDiscoverServerAsync();
            if (!string.IsNullOrEmpty(discovered))
            {
                var url = GetFullUrl("/api/v1/presets");
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();
                var json = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<PresetListResponse>(json);
                if (result?.Presets != null)
                {
                    foreach (var p in result.Presets)
                    {
                        if (!string.IsNullOrEmpty(p.CoverUrl))
                        {
                            p.FullCoverUrl = GetFullUrl(p.CoverUrl);
                        }
                    }
                    return result.Presets;
                }
            }
            throw;
        }
    }

    public async Task<FaceAnalysisResponse?> AnalyzeFaceAsync(Stream imageStream, string fileName)
    {
        try
        {
            var url = GetFullUrl("/api/v1/analyze");
            using var content = new MultipartFormDataContent();

            var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var response = await _httpClient.PostAsync(url, content);
            var json = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError("Analyze failed ({Status}): {Body}", response.StatusCode, json);
                throw new HttpRequestException($"Ошибка анализа ({response.StatusCode}): {json}");
            }

            return JsonSerializer.Deserialize<FaceAnalysisResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to analyze face");
            throw;
        }
    }

    public async Task<TaskResponse?> StartGenerationAsync(Stream imageStream, string fileName, GenerationParams parameters)
    {
        try
        {
            var url = GetFullUrl("/api/v1/generate");
            using var content = new MultipartFormDataContent();

            var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            // Параметры формы (с защитой от null)
            content.Add(new StringContent(parameters.PresetId ?? string.Empty), "preset_id");
            content.Add(new StringContent(parameters.HairMode ?? "keep"), "hair_mode");

            if (!string.IsNullOrWhiteSpace(parameters.HairStyle))
                content.Add(new StringContent(parameters.HairStyle), "hair_style");

            if (!string.IsNullOrWhiteSpace(parameters.HairColor))
                content.Add(new StringContent(parameters.HairColor), "hair_color");

            content.Add(new StringContent(parameters.Ethnicity ?? "auto"), "ethnicity");
            content.Add(new StringContent(parameters.Gender ?? "auto"), "gender");
            content.Add(new StringContent(parameters.BodyBuild ?? "auto"), "body_build");

            content.Add(new StringContent(parameters.FaceStrength.ToString(CultureInfo.InvariantCulture)), "face_strength");
            content.Add(new StringContent(parameters.Steps.ToString()), "steps");
            content.Add(new StringContent(parameters.GuidanceScale.ToString(CultureInfo.InvariantCulture)), "guidance_scale");

            content.Add(new StringContent(parameters.ApplyPreEnhance.ToString().ToLowerInvariant()), "apply_pre_enhance");
            content.Add(new StringContent(parameters.ApplyFaceSwap.ToString().ToLowerInvariant()), "apply_face_swap");
            content.Add(new StringContent(parameters.ApplyFaceRestore.ToString().ToLowerInvariant()), "apply_face_restore");
            content.Add(new StringContent(parameters.FidelityWeight.ToString(CultureInfo.InvariantCulture)), "fidelity_weight");
            content.Add(new StringContent(parameters.ApplyFilmGrain.ToString().ToLowerInvariant()), "apply_film_grain");

            var response = await _httpClient.PostAsync(url, content);
            var json = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError("Generate failed ({Status}): {Body}", response.StatusCode, json);
                throw new HttpRequestException($"Ошибка запуска генерации ({response.StatusCode}): {json}");
            }

            var task = JsonSerializer.Deserialize<TaskResponse>(json);
            if (task != null)
            {
                PopulateFullUrls(task);
            }
            return task;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start generation");
            throw;
        }
    }

    public async Task<TaskResponse?> GetTaskStatusAsync(string taskId)
    {
        try
        {
            var url = GetFullUrl($"/api/v1/tasks/{taskId}");
            var response = await _httpClient.GetAsync(url);
            var json = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError("Task status failed ({Status}): {Body}", response.StatusCode, json);
                throw new HttpRequestException($"Ошибка проверки статуса ({response.StatusCode}): {json}");
            }

            var task = JsonSerializer.Deserialize<TaskResponse>(json);
            if (task != null)
            {
                PopulateFullUrls(task);
            }
            return task;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get task status for {TaskId}", taskId);
            throw;
        }
    }

    private void PopulateFullUrls(TaskResponse task)
    {
        if (!string.IsNullOrEmpty(task.ResultUrl))
        {
            task.FullResultUrl = GetFullUrl(task.ResultUrl);
        }
        if (!string.IsNullOrEmpty(task.PreviewFaceUrl))
        {
            task.FullFaceUrl = GetFullUrl(task.PreviewFaceUrl);
        }
    }
}
