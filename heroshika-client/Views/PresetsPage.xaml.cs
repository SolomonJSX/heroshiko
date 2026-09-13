using heroshika_client.ViewModels;

namespace heroshika_client.Views;

public partial class PresetsPage : ContentPage
{
    private readonly PresetsViewModel _viewModel;

    public PresetsPage(PresetsViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = _viewModel = viewModel;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        _ = Task.Run(async () =>
        {
            try
            {
                await _viewModel.InitializeAsync();
            }
            catch
            {
                // Игнорируем сетевые сбои в фоне, UI не должен зависать
            }
        });
    }
}
