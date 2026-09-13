using heroshika_client.ViewModels;

namespace heroshika_client.Views;

public partial class GenerationProgressPage : ContentPage
{
    private readonly GenerationProgressViewModel _viewModel;

    public GenerationProgressPage(GenerationProgressViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = _viewModel = viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await _viewModel.StartProcessAsync();
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _viewModel.CancelPolling();
    }
}
