using heroshika_client.Views;

namespace heroshika_client;

public partial class AppShell : Shell
{
    public AppShell()
    {
        InitializeComponent();

        Routing.RegisterRoute("FaceAnalysisPage", typeof(FaceAnalysisPage));
        Routing.RegisterRoute("GenerationTuningPage", typeof(GenerationTuningPage));
        Routing.RegisterRoute("GenerationProgressPage", typeof(GenerationProgressPage));
        Routing.RegisterRoute("SettingsPage", typeof(SettingsPage));
    }
}
