using heroshika_client.ViewModels;

namespace heroshika_client.Views;

public partial class FaceAnalysisPage : ContentPage
{
    private readonly FaceAnalysisViewModel _viewModel;

    public FaceAnalysisPage(FaceAnalysisViewModel viewModel)
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
