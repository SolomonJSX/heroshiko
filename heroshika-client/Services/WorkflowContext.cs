using heroshika_client.Models;

namespace heroshika_client.Services;

public interface IWorkflowContext
{
    PresetResponse? SelectedPreset { get; set; }
    string? SelectedPhotoPath { get; set; }
    byte[]? SelectedPhotoBytes { get; set; }
    FaceAnalysisResponse? AnalysisResult { get; set; }
    GenerationParams GenerationParams { get; set; }
    TaskResponse? CurrentTask { get; set; }
    void Reset();
}

public class WorkflowContext : IWorkflowContext
{
    public PresetResponse? SelectedPreset { get; set; }
    public string? SelectedPhotoPath { get; set; }
    public byte[]? SelectedPhotoBytes { get; set; }
    public FaceAnalysisResponse? AnalysisResult { get; set; }
    public GenerationParams GenerationParams { get; set; } = new();
    public TaskResponse? CurrentTask { get; set; }

    public void Reset()
    {
        SelectedPreset = null;
        SelectedPhotoPath = null;
        SelectedPhotoBytes = null;
        AnalysisResult = null;
        GenerationParams = new GenerationParams();
        CurrentTask = null;
    }
}
