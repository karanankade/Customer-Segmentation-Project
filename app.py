from flask import Flask, jsonify, render_template_string, send_from_directory
import os
import pandas as pd

app = Flask(__name__)

# Run segmentation logic if output directory or files don't exist
def ensure_segmentation_data():
    if not os.path.exists('output/cluster_profiles.csv') or not os.path.exists('customers.csv'):
        try:
            import generate_data
            generate_data.main()
        except Exception as e:
            print("Error generating data:", e)
        try:
            import segmentation
            segmentation.main()
        except Exception as e:
            print("Error running segmentation:", e)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Segmentation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .hero { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 2.5rem 0; margin-bottom: 2rem; border-radius: 0 0 1rem 1rem; }
        .card { border: none; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
        .card-header { background-color: white; border-bottom: 1px solid #e5e7eb; font-weight: 600; }
        .table-custom { border-radius: 0.5rem; overflow: hidden; }
        .badge-cluster { font-size: 0.9rem; padding: 0.4em 0.8em; }
    </style>
</head>
<body>
    <div class="hero text-center">
        <div class="container">
            <h1 class="display-5 fw-bold">Customer Segmentation Dashboard</h1>
            <p class="lead mb-0">Unsupervised Machine Learning with K-Means & PCA</p>
        </div>
    </div>

    <div class="container">
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>Cluster Profiles Summary</span>
                        <a href="/api/clusters" class="btn btn-sm btn-outline-primary" target="_blank">View Raw JSON API</a>
                    </div>
                    <div class="card-body">
                        {% if profile_html %}
                            <div class="table-responsive">
                                {{ profile_html | safe }}
                            </div>
                        {% else %}
                            <div class="alert alert-warning mb-0">Segment data is generating... Please refresh in a moment.</div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Customer Segments (PCA 2D Plot)</div>
                    <div class="card-body text-center">
                        <img src="/output/cluster_pca.png" class="img-fluid rounded" alt="PCA Scatter Plot" onerror="this.src='https://via.placeholder.com/500x350?text=Plot+Generating...'">
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Elbow Method Curve (Optimal K)</div>
                    <div class="card-body text-center">
                        <img src="/output/elbow_curve.png" class="img-fluid rounded" alt="Elbow Curve" onerror="this.src='https://via.placeholder.com/500x350?text=Plot+Generating...'">
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    ensure_segmentation_data()
    profile_html = ""
    profile_path = 'output/cluster_profiles.csv'
    if os.path.exists(profile_path):
        df_profile = pd.read_csv(profile_path)
        profile_html = df_profile.to_html(classes="table table-hover table-bordered table-custom text-center mb-0", index=False)
    return render_template_string(HTML_TEMPLATE, profile_html=profile_html)

@app.route('/api/clusters')
def api_clusters():
    ensure_segmentation_data()
    profile_path = 'output/cluster_profiles.csv'
    if os.path.exists(profile_path):
        df_profile = pd.read_csv(profile_path)
        return jsonify(df_profile.to_dict(orient='records'))
    return jsonify({"status": "processing", "message": "Cluster data is generating"}), 202

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    ensure_segmentation_data()
    app.run(debug=True, port=5000)
