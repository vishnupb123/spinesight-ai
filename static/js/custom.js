// static/js/custom.js

document.addEventListener("DOMContentLoaded", function() {
    const inputElements = document.querySelectorAll('.feature-input');
    const ctx = document.getElementById('featureChart').getContext('2d');
  
    const medicalThresholds = {
        pelvic_incidence: [40, 70],
        pelvic_tilt: [5, 25],
        lumbar_lordosis_angle: [20, 60],
        sacral_slope: [30, 50],
        pelvic_radius: [90, 130],
        degree_spondylolisthesis: [0, 10]
    };
  
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(medicalThresholds),
            datasets: [{
                label: 'Feature Value',
                data: Object.keys(medicalThresholds).map(() => 0),
                backgroundColor: Object.keys(medicalThresholds).map(() => '#28a745')
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
  
    inputElements.forEach((input, index) => {
        input.addEventListener('input', () => {
            const value = parseFloat(input.value) || 0;
            const feature = input.name;
            const [min, max] = medicalThresholds[feature];
            chart.data.datasets[0].data[index] = value;
            chart.data.datasets[0].backgroundColor[index] = (value < min || value > max) ? '#dc3545' : '#28a745';
            chart.update();
        });
    });
  
    // Model selection dynamic update (if needed)
    const modelSelect = document.getElementById('model-select');
    const modelInfoText = document.getElementById('model-info-text');
    modelSelect.addEventListener('change', function () {
        let info = '';
        if (this.value === 'lr') {
            info = 'Logistic Regression is a simple and interpretable model used for classification tasks.';
        } else if (this.value === 'svc') {
            info = 'Support Vector Classifier is robust and effective for high-dimensional data.';
        } else if (this.value === 'rf') {
            info = 'Random Forest is a powerful ensemble method that provides feature importance insights.';
        }
        modelInfoText.textContent = info;
    });
  });
  