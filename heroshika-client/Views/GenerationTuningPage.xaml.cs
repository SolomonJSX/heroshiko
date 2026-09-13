using heroshika_client.ViewModels;

namespace heroshika_client.Views;

public partial class GenerationTuningPage : ContentPage
{
    private readonly GenerationTuningViewModel _viewModel;

    public GenerationTuningPage(GenerationTuningViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = _viewModel = viewModel;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        _viewModel.Initialize();
    }
}
