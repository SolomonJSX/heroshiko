using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using heroshika_client.Models;
using heroshika_client.Services;
using Microsoft.Extensions.Logging;

namespace heroshika_client.ViewModels;

public class TuningOption
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public override string ToString() => Title;
}

public partial class GenerationTuningViewModel : BaseViewModel
{
    private readonly IWorkflowContext _workflowContext;
    private readonly ILogger<GenerationTuningViewModel> _logger;

    [ObservableProperty]
    public partial string PresetTitle { get; set; } = string.Empty;

    // Режим волос: "keep" (моя), "change" (сменить), "preset" (из стиля)
    [ObservableProperty]
    public partial string HairMode { get; set; } = "keep";

    [ObservableProperty]
    public partial bool IsHairKeep { get; set; } = true;

    [ObservableProperty]
    public partial bool IsHairChange { get; set; }

    [ObservableProperty]
    public partial bool IsHairPreset { get; set; }

    [ObservableProperty]
    public partial string SelectedHairStyle { get; set; } = "Короткая модельная";

    [ObservableProperty]
    public partial string SelectedHairColor { get; set; } = "Шоколадный / Каштановый";

    [ObservableProperty]
    public partial TuningOption? SelectedEthnicityOption { get; set; }

    [ObservableProperty]
    public partial TuningOption? SelectedGenderOption { get; set; }

    [ObservableProperty]
    public partial TuningOption? SelectedBodyBuildOption { get; set; }

    [ObservableProperty]
    public partial float FaceStrength { get; set; } = 0.40f;

    [ObservableProperty]
    public partial int Steps { get; set; } = 25;

    [ObservableProperty]
    public partial float GuidanceScale { get; set; } = 5.5f;

    [ObservableProperty]
    public partial bool ApplyPreEnhance { get; set; } = false;

    [ObservableProperty]
    public partial bool ApplyFaceSwap { get; set; } = true;

    [ObservableProperty]
    public partial bool ApplyFaceRestore { get; set; } = true;

    [ObservableProperty]
    public partial float FidelityWeight { get; set; } = 0.80f;

    [ObservableProperty]
    public partial bool ApplyFilmGrain { get; set; } = true;

    [ObservableProperty]
    public partial bool ShowAdvanced { get; set; }

    [ObservableProperty]
    public partial string AdvancedToggleText { get; set; } = "Показать ▼";

    public List<string> HairStyles { get; } = new()
    {
        "Короткая модельная",
        "Каре / Боб",
        "Классический хвост / Пучок",
        "Объемные локоны / Волны",
        "Длинные шелковистые прямые",
        "Текстурированный фейд (Fade)",
        "Корейский стиль (Two-Block)"
    };

    public List<string> HairColors { get; } = new()
    {
        "Натуральный темный / Брюнет",
        "Шоколадный / Каштановый",
        "Натуральный русый",
        "Теплый блонд",
        "Пепельно-платиновый",
        "Медно-рыжий"
    };

    public List<TuningOption> Ethnicities { get; } = new()
    {
        new TuningOption { Id = "auto", Title = "Автоматически (по биометрии лица)" },
        new TuningOption { Id = "central_asian", Title = "Центральная Азия (Казахстан / Тюркский)" },
        new TuningOption { Id = "european", Title = "Европейский (Светлый тип)" },
        new TuningOption { Id = "east_asian", Title = "Восточная Азия (Корейский / Японский)" },
        new TuningOption { Id = "middle_eastern", Title = "Ближневосточный (Средиземноморье)" },
        new TuningOption { Id = "latino", Title = "Латиноамериканский" }
    };

    public List<TuningOption> Genders { get; } = new()
    {
        new TuningOption { Id = "man", Title = "👨 Мужской (Парень / Мужчина)" },
        new TuningOption { Id = "woman", Title = "👩 Женский (Девушка / Женщина)" },
        new TuningOption { Id = "auto", Title = "⚡ Автоматически (по биометрии лица)" }
    };

    public List<TuningOption> BodyBuilds { get; } = new()
    {
        new TuningOption { Id = "auto", Title = "Автоматически (по биометрии лица)" },
        new TuningOption { Id = "regular", Title = "Естественное / Стандартное" },
        new TuningOption { Id = "slender", Title = "Стройное / Модельное" },
        new TuningOption { Id = "athletic", Title = "Спортивное / Атлетичное" },
        new TuningOption { Id = "plus_size", Title = "Плотное / Plus Size" }
    };

    public GenerationTuningViewModel(
        IWorkflowContext workflowContext,
        ILogger<GenerationTuningViewModel> logger)
    {
        _workflowContext = workflowContext;
        _logger = logger;
        Title = "Настройка стиля";

        SelectedGenderOption = Genders[0]; // По умолчанию Мужской
        SelectedEthnicityOption = Ethnicities[0];
        SelectedBodyBuildOption = BodyBuilds[0];
    }

    [RelayCommand]
    public void Initialize()
    {
        PresetTitle = _workflowContext.SelectedPreset?.Title ?? "Стиль";
        var analysis = _workflowContext.AnalysisResult;

        if (analysis != null && !string.IsNullOrEmpty(analysis.Gender))
        {
            var matchGender = Genders.FirstOrDefault(g => g.Id.Equals(analysis.Gender, StringComparison.OrdinalIgnoreCase));
            if (matchGender != null)
            {
                SelectedGenderOption = matchGender;
            }
        }

        if (analysis != null && !string.IsNullOrEmpty(analysis.BuildType))
        {
            var match = BodyBuilds.FirstOrDefault(b => b.Id.Equals(analysis.BuildType, StringComparison.OrdinalIgnoreCase));
            if (match != null)
            {
                SelectedBodyBuildOption = match;
            }
        }
    }

    partial void OnHairModeChanged(string value)
    {
        IsHairKeep = value == "keep";
        IsHairChange = value == "change";
        IsHairPreset = value == "preset";
    }

    [RelayCommand]
    public void SetHairMode(string mode)
    {
        HairMode = mode;
    }

    [RelayCommand]
    public void ToggleAdvanced()
    {
        ShowAdvanced = !ShowAdvanced;
        AdvancedToggleText = ShowAdvanced ? "Скрыть ▲" : "Показать ▼";
    }

    [RelayCommand]
    public async Task StartPhotoshootAsync()
    {
        var ethnicity = SelectedEthnicityOption?.Id ?? "auto";
        var gender = SelectedGenderOption?.Id ?? "man";
        var bodyBuild = SelectedBodyBuildOption?.Id ?? "auto";
        var presetId = _workflowContext.SelectedPreset?.Id ?? "dubai_luxury";

        _workflowContext.GenerationParams = new GenerationParams
        {
            PresetId = presetId,
            HairMode = HairMode ?? "keep",
            HairStyle = IsHairChange ? SelectedHairStyle : null,
            HairColor = IsHairChange ? SelectedHairColor : null,
            Ethnicity = ethnicity,
            Gender = gender,
            BodyBuild = bodyBuild,
            FaceStrength = FaceStrength,
            Steps = Steps,
            GuidanceScale = GuidanceScale,
            ApplyPreEnhance = ApplyPreEnhance,
            ApplyFaceSwap = ApplyFaceSwap,
            ApplyFaceRestore = ApplyFaceRestore,
            FidelityWeight = FidelityWeight,
            ApplyFilmGrain = ApplyFilmGrain
        };

        await Shell.Current.GoToAsync("GenerationProgressPage");
    }

    [RelayCommand]
    public async Task GoBackAsync()
    {
        await Shell.Current.GoToAsync("..");
    }
}
