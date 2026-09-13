using System.Text.Json.Serialization;

namespace heroshika_client.Models;

public enum TaskStatus
{
    [JsonPropertyName("queued")]
    Queued,

    [JsonPropertyName("processing")]
    Processing,

    [JsonPropertyName("completed")]
    Completed,

    [JsonPropertyName("failed")]
    Failed
}

public class TaskResponse
{
    [JsonPropertyName("task_id")]
    public string TaskId { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = "queued";

    [JsonPropertyName("progress")]
    public int Progress { get; set; }

    [JsonPropertyName("preset_id")]
    public string PresetId { get; set; } = string.Empty;

    [JsonPropertyName("result_url")]
    public string? ResultUrl { get; set; }

    [JsonPropertyName("preview_face_url")]
    public string? PreviewFaceUrl { get; set; }

    [JsonPropertyName("error_message")]
    public string? ErrorMessage { get; set; }

    [JsonPropertyName("elapsed_seconds")]
    public float? ElapsedSeconds { get; set; }

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; } = new();

    [JsonIgnore]
    public bool IsCompleted => Status.Equals("completed", StringComparison.OrdinalIgnoreCase);

    [JsonIgnore]
    public bool IsFailed => Status.Equals("failed", StringComparison.OrdinalIgnoreCase);

    [JsonIgnore]
    public bool IsInProgress => Status.Equals("queued", StringComparison.OrdinalIgnoreCase) ||
                               Status.Equals("processing", StringComparison.OrdinalIgnoreCase);

    [JsonIgnore]
    public string FullResultUrl { get; set; } = string.Empty;

    [JsonIgnore]
    public string FullFaceUrl { get; set; } = string.Empty;
}
