using heroshika_client.Services;
using heroshika_client.ViewModels;
using heroshika_client.Views;
using Microsoft.Extensions.Logging;

namespace heroshika_client;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
                fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
            });

        // 1. HTTP Client & Services
        builder.Services.AddHttpClient();
        builder.Services.AddSingleton<ISettingsService, SettingsService>();
        builder.Services.AddSingleton<IWorkflowContext, WorkflowContext>();
        builder.Services.AddSingleton<IHeroshikoApiService, HeroshikoApiService>();

        // 2. ViewModels
        builder.Services.AddTransient<PresetsViewModel>();
        builder.Services.AddTransient<FaceAnalysisViewModel>();
        builder.Services.AddTransient<GenerationTuningViewModel>();
        builder.Services.AddTransient<GenerationProgressViewModel>();
        builder.Services.AddTransient<SettingsViewModel>();

        // 3. Views
        builder.Services.AddTransient<PresetsPage>();
        builder.Services.AddTransient<FaceAnalysisPage>();
        builder.Services.AddTransient<GenerationTuningPage>();
        builder.Services.AddTransient<GenerationProgressPage>();
        builder.Services.AddTransient<SettingsPage>();

#if DEBUG
        builder.Logging.AddDebug();
#endif

        return builder.Build();
    }
}
