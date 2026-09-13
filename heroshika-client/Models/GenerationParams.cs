namespace heroshika_client.Models;

public class GenerationParams
{
    public string PresetId { get; set; } = string.Empty;
    public string HairMode { get; set; } = "keep"; // "keep", "change", "preset"
    public string? HairStyle { get; set; }
    public string? HairColor { get; set; }
    public string Ethnicity { get; set; } = "auto";
    public string Gender { get; set; } = "auto"; // "auto", "man", "woman"
    public string BodyBuild { get; set; } = "auto";
    public float FaceStrength { get; set; } = 0.40f;
    public int Steps { get; set; } = 25;
    public float GuidanceScale { get; set; } = 5.5f;
    public bool ApplyPreEnhance { get; set; } = false;
    public bool ApplyFaceSwap { get; set; } = true;
    public bool ApplyFaceRestore { get; set; } = true;
    public float FidelityWeight { get; set; } = 0.80f;
    public bool ApplyFilmGrain { get; set; } = true;
}
