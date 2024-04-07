from flask import Flask, render_template, request, send_file, redirect, url_for
import numpy as np
import iisignature

app = Flask(__name__)

def compute_signatures(paths, order):
    return [iisignature.sig(path, order) for path in paths]

def expected_signature(paths, order):
    signatures = compute_signatures(paths, order)
    return np.mean(signatures, axis=0)

# Function to generate geometric Brownian motion paths
def generate_geometric_brownian_motion_paths(num_paths=1, num_steps=100, mu=0, sigma=1):
    dt = 1.0 / num_steps  # time step
    times = np.linspace(0, 1, num_steps + 1)  # Time grid

    # Generate Brownian increments
    dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, num_steps))
    # Cumulative sum of increments to simulate Brownian motion
    W = np.cumsum(dW, axis=1)
    # Prepend zeros for the initial value of Brownian motion
    W = np.hstack([np.zeros((num_paths, 1)), W])

    # Calculate the paths using the GBM formula
    paths = np.exp((mu - 0.5 * sigma**2) * times + sigma * W)

    # Initialize the array
    output = np.zeros((num_paths, num_steps + 1, 2))
    # Fill in the time grid for all paths
    output[:, :, 0] = np.tile(times, (num_paths, 1))
    # Fill in the paths
    output[:, :, 1] = paths

    return output

# Handle index route
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/generate', methods=['POST'])
def generate():
    # Get number of data points from form
    num_points = int(request.form['num_points'])

    # Constants for simulation
    num_paths = num_points
    num_steps = 100
    signature_order = 2

    # Lists to store features and targets for modeling
    features_list = []
    targets_list = []

    # Generate data points
    for mu in np.arange(0, 1.1, 0.01):
        for sigma in np.arange(0, 1.1, 0.01):
            # Generate geometric Brownian motion paths
            gb_paths = generate_geometric_brownian_motion_paths(num_paths, num_steps, mu, sigma)
            expected_sig = expected_signature(gb_paths, signature_order)
        
            # Select specific elements of expected signature for model features
            selected_features = np.array([expected_sig[1], expected_sig[3], expected_sig[4], expected_sig[5]])
        
            # Reshape and store selected features
            selected_features_reshaped = selected_features.reshape(1, -1)
            features_list.append(selected_features_reshaped)
        
            # Define model targets based on mu rate and sigma
            target_values = np.array([[mu, sigma]])
            targets_list.append(target_values)

    # Concatenate feature and target arrays
    features_matrix = np.concatenate(features_list, axis=0)
    targets_matrix = np.concatenate(targets_list, axis=0)

    # Prepare CSV files
    features_filename = "static/features_matrix.csv"
    targets_filename = "static/targets_matrix.csv"
    np.savetxt(features_filename, features_matrix, delimiter=',')
    np.savetxt(targets_filename, targets_matrix, delimiter=',')

    # Redirect to download page
    return redirect(url_for('download_page'))

# Handle download page
@app.route('/download')
def download_page():
    return render_template('download.html')

if __name__ == '__main__':
    app.run(debug=True)
