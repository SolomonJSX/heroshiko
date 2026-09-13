using System.Text.Json.Serialization;

namespace heroshika_client.Models;

public class PresetResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = string.Empty;

    [JsonPropertyName("negative_prompt")]
    public string NegativePrompt { get; set; } = string.Empty;

    [JsonPropertyName("control_strength")]
    public float ControlStrength { get; set; } = 0.75f;

    [JsonPropertyName("denoising_strength")]
    public float DenoisingStrength { get; set; } = 0.65f;

    [JsonPropertyName("cover_image")]
    public string? CoverImage { get; set; }

    [JsonPropertyName("cover_url")]
    public string? CoverUrl { get; set; }

    [JsonPropertyName("preview_url")]
    public string? PreviewUrl { get; set; }

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new();

    [JsonIgnore]
    public string FullCoverUrl { get; set; } = string.Empty;

    // Автоматический выбор обложки (надежные локальные ассеты или серверный URL)
    [JsonIgnore]
    public string DisplayCoverSource
    {
        get
        {
            var id = Id?.ToLowerInvariant() ?? string.Empty;
            if (id.Contains("dubai")) return "dubai_luxury.jpg";
            if (id.Contains("paris")) return "paris_cafe.jpg";
            if (id.Contains("group")) return "kpop_group.jpg";
            if (id.Contains("kpop") || id.Contains("idol")) return "kpop_idol.jpg";

            if (!string.IsNullOrEmpty(FullCoverUrl))
                return FullCoverUrl;

            return "heroshiko_icon.png";
        }
    }
}

public class PresetListResponse
{
    [JsonPropertyName("presets")]
    public List<PresetResponse> Presets { get; set; } = new();

    [JsonPropertyName("total")]
    public int Total { get; set; }
}
