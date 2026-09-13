using System.Text.Json.Serialization;

namespace heroshika_client.Models;

public class FaceAnalysisResponse
{
    [JsonPropertyName("face_detected")]
    public bool FaceDetected { get; set; }

    [JsonPropertyName("multiple_faces_detected")]
    public bool MultipleFacesDetected { get; set; }

    [JsonPropertyName("total_faces_count")]
    public int TotalFacesCount { get; set; } = 1;

    [JsonPropertyName("gender")]
    public string? Gender { get; set; }

    [JsonPropertyName("age")]
    public int? Age { get; set; }

    [JsonPropertyName("hair_color")]
    public string? HairColor { get; set; }

    [JsonPropertyName("hair_length")]
    public string? HairLength { get; set; }

    [JsonPropertyName("has_beard")]
    public bool HasBeard { get; set; }

    [JsonPropertyName("hair_prompt")]
    public string? HairPrompt { get; set; }

    [JsonPropertyName("build_type")]
    public string? BuildType { get; set; }

    [JsonPropertyName("build_label_ru")]
    public string? BuildLabelRu { get; set; }

    [JsonPropertyName("bbox")]
    public List<int>? Bbox { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = "Успешно";

    [JsonIgnore]
    public string GenderRu => Gender?.ToLowerInvariant() switch
    {
        "male" or "man" => "Мужчина",
        "female" or "woman" => "Женщина",
        _ => Gender ?? "Не определен"
    };

    [JsonIgnore]
    public string SummaryText =>
        FaceDetected
            ? $"{GenderRu}, {Age} лет | Волосы: {HairColor ?? "темные"} ({HairLength ?? "средние"}) | Комплекция: {BuildLabelRu ?? "обычная"}"
            : "Лицо не обнаружено";
}
