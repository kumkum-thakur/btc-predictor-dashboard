document.addEventListener("DOMContentLoaded", function () {
    const daysInput = document.getElementById("daysInput");
    const lstmBtn = document.getElementById("lstmBtn");
    const arimaBtn = document.getElementById("arimaBtn");
    const recommendationDiv = document.getElementById("recommendation");
    const ctx = document.getElementById("forecastChart").getContext("2d");
    const minVal = document.getElementById("minVal");
    const maxVal = document.getElementById("maxVal");
    const avgVal = document.getElementById("avgVal");

    let chart;

    function updateChart(data) {
        const labels = data.map(point => point[0]);
        const values = data.map(point => point[1]);

        if (chart) chart.destroy();

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Forecasted BTC Price (USD)',
                    data: values,
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (USD)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: 'white'
                        }
                    }
                }
            }
        });

        updateStats(values);
        updateRecommendation(values);
    }

    function updateStats(values) {
        const min = Math.min(...values).toFixed(2);
        const max = Math.max(...values).toFixed(2);
        const avg = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);

        minVal.innerText = `$${min}`;
        maxVal.innerText = `$${max}`;
        avgVal.innerText = `$${avg}`;
    }

    function updateRecommendation(values) {
    if (values.length < 2) {
        recommendationDiv.innerText = "⚠️ Not enough data to analyze.";
        return;
    }

    const start = values[0];
    const end = values[values.length - 1];
    const change = ((end - start) / start) * 100;

    console.log("Start:", start, "End:", end, "Change:", change.toFixed(2));

    let message = "";
    if (change > 2) {
        message = `📈 BTC expected to rise by ${change.toFixed(2)}% — Consider Buying!`;
    } else if (change < -2) {
        message = `📉 BTC may fall by ${Math.abs(change).toFixed(2)}% — Consider Selling!`;
    } else {
        message = `📊 BTC likely stable (${change.toFixed(2)}%) — Hold.`;
    }

    recommendationDiv.innerText = message;
}



    function fetchForecast(model) {
        const days = daysInput.value || 7;
        recommendationDiv.innerText = "⏳ Loading forecast...";

        fetch(`/predict/${model}?days=${days}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.forecast) {
                    updateChart(data.forecast);
                } else {
                    recommendationDiv.innerText = "⚠️ Something went wrong with the forecast.";
                }
            })
            .catch(error => {
                console.error("Error fetching forecast:", error);
                recommendationDiv.innerText = "❌ Error loading forecast. Please try again.";
            });
    }

    lstmBtn.addEventListener("click", () => fetchForecast("lstm"));
    arimaBtn.addEventListener("click", () => fetchForecast("arima"));

    console.log("script.js loaded ✅");
});
