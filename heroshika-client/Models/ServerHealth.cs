using System.Text.Json.Serialization;

namespace heroshika_client.Models;

public class ServerHealthResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "unknown";

    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;

    [JsonPropertyName("cuda_available")]
    public bool CudaAvailable { get; set; }

    [JsonPropertyName("device_name")]
    public string? DeviceName { get; set; }

    [JsonPropertyName("vram_allocated_mb")]
    public float? VramAllocatedMb { get; set; }

    [JsonPropertyName("vram_reserved_mb")]
    public float? VramReservedMb { get; set; }

    [JsonPropertyName("vram_total_mb")]
    public float? VramTotalMb { get; set; }

    [JsonPropertyName("models_loaded")]
    public List<string> ModelsLoaded { get; set; } = new();

    [JsonIgnore]
    public bool IsOnline => Status.Equals("ok", StringComparison.OrdinalIgnoreCase);

    [JsonIgnore]
    public string VramSummary =>
        VramTotalMb.HasValue && VramAllocatedMb.HasValue
            ? $"{VramAllocatedMb:F0} / {VramTotalMb:F0} MB"
            : "N/A";
}
