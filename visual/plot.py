import matplotlib.pyplot as plt

def plot_forecast(forecast_dict, variable):
    values = forecast_dict.get(variable, [])
    if not values:
        print(f"No data found for variable '{variable}'")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, marker='o')
    plt.title(f"Forecast for {variable}")
    plt.xlabel("6-hour steps")
    plt.ylabel(variable)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
