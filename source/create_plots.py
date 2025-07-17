import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import features


def get_g3_histogram(subject, school):
    """
    Create histograms of G3 for selected subject and school.

    Parameters:
        subject (str): subject name ('por' or 'mat')
        school (str): school name ('GP' or 'MS')
    """

    df = pd.read_csv(f"../data/student-{subject}.csv", sep=';')

    df = df[df['school'] == school]

    plt.figure(figsize=(8, 5))
    bins = [i - 0.5 for i in range(0,22)]
    plt.hist(df['G3'], bins=bins, edgecolor='black', color='skyblue')
    plt.xticks(range(0, 21, 1))
    #plt.title(f"Distribution of Final Grade (G3) for {subject} in {school} School")
    plt.xlabel('Final Grade (G3)')
    plt.ylabel('Number of students')
    plt.grid(axis='y', linestyle='--', alpha=0.75)
    plt.tight_layout()

    plt.savefig(f"../plots/{subject}/g3_histogram_{school}.png", dpi=300)
    plt.show()
    plt.close()

def get_absences_plots(subject):
    """
    Create feature plots before and after scaling for a selected subject.

    Parameters:
        subject (str): subject name ('por' or 'mat')
    """

    df = pd.read_csv(f"../data/student-{subject}.csv", sep=';')

    absences_raw = df[['absences']].values
    absences_scaled = StandardScaler().fit_transform(absences_raw)

    make_unscaled_abs(absences_raw, subject)
    make_scaled_abs(absences_scaled, subject)

def make_unscaled_abs(df, subject):
    """
    Helper function for creating unscaled absences plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing absences data.
        subject (str): subject name ('por' or 'mat')
    """
    plt.figure(figsize=(8, 5))


    plt.hist(df, bins='auto', color='skyblue', edgecolor='black')
    #plt.title("Unscaled Absences")
    plt.xlabel("Absences")
    plt.ylabel("Number of students")
    plt.grid(axis='y', linestyle='--', alpha=0.75)
    plt.tight_layout()

    plt.savefig(f"../plots/{subject}/unscaled_abs.png", dpi=300)
    plt.show()
    plt.close()


def make_scaled_abs(df, subject):
    """
        Helper function for creating scaled absences plot.

        Parameters:
            df (pd.DataFrame): DataFrame containing absences data.
            subject (str): subject name ('por' or 'mat')
        """
    plt.figure(figsize=(8, 5))

    plt.hist(df, bins='auto', color='orange', edgecolor='black')
    #plt.title("Standard Scaled Absences")
    plt.xlabel("Absences")
    plt.ylabel("Number of students")
    plt.grid(axis='y', linestyle='--', alpha=0.75)
    plt.tight_layout()

    plt.savefig(f"../plots/{subject}/scaled_abs.png", dpi=300)
    plt.show()
    plt.close()

def make_poly_features(subject):
    """
    Create a plot of bars to compare number of original features and number of features when PolynomialFeatures()
    is used with interaction_only=True and interaction_only=False.

    Parameters:
         subject (str): subject name ('por' or 'mat')
    """

    df = pd.read_csv(f"../data/student-{subject}.csv", sep=';')

    df_num = df[features.FeatureList().numerical]

    # Count original features
    num_original_features = df_num.shape[1]

    # Generate polynomial features
    poly1 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features1 = poly1.fit_transform(df_num)

    # Get number of polynomial features
    num_poly_features1 = poly_features1.shape[1]

    # Generate polynomial features
    poly2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features2 = poly2.fit_transform(df_num)

    # Get number of polynomial features
    num_poly_features2 = poly_features2.shape[1]

    # Plot comparison
    plt.figure(figsize=(12, 7))
    plt.bar(['Original', 'Polynomial (deg=2, interaction_only=True)',
             'Polynomial (deg=2, interaction_only=False)'], [num_original_features, num_poly_features1,
                                                             num_poly_features2], color=['skyblue', 'orange', 'green'], edgecolor='black')
    plt.ylabel('Number of features')

    for i, v in enumerate([num_original_features, num_poly_features1, num_poly_features2]):
        plt.text(i, v, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"../plots/{subject}/feature_num_bar.png", dpi=300)
    plt.close()


def plot_feature_importance(csv_filepath, title, output_filename):
    """
    Reads a model's coefficient CSV file and generates a feature importance plot.

    Args:
        csv_filepath (str): The path to the input coefficient CSV file.
        title (str): The title for the plot.
        output_filename (str): The filename to save the plot as (e.g., 'plot.png').
    """
    # Read the wide-format CSV
    coeffs_df = pd.read_csv(csv_filepath)

    # Transpose to long format (Feature, Coefficient)
    coeffs_long = coeffs_df.T.reset_index()
    coeffs_long.columns = ['feature', 'coefficient']

    # Separate the intercept term
    feature_coeffs = coeffs_long[coeffs_long['feature'] != 'intercept'].copy()

    # Convert coefficients to a numeric type
    feature_coeffs['Coefficient'] = pd.to_numeric(feature_coeffs['coefficient'], errors='coerce')
    feature_coeffs.dropna(subset=['Coefficient'], inplace=True)

    # For Lasso/Elastic Net, filter out zeroed-out coefficients
    if 'lasso' in csv_filepath or 'elastic_net' in csv_filepath:
        feature_coeffs = feature_coeffs[feature_coeffs['coefficient'] != 0]

    # Sort features by coefficient value for plotting
    plot_coeffs = feature_coeffs.sort_values(by='coefficient')

    plt.style.use('seaborn-v0_8-whitegrid')

    # Adjust figure height dynamically based on the number of features
    num_features = len(plot_coeffs)
    fig_height = max(8, num_features * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Color bars blue for positive and red for negative coefficients
    colors = ['blue' if c > 0 else 'red' for c in plot_coeffs['coefficient']]

    ax.barh(plot_coeffs['feature'], plot_coeffs['coefficient'], color=colors)

    # Add labels and formatting
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Coefficient value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=14)
    plt.yticks(fontsize=12)
    ax.axvline(0, color='grey', linewidth=0.8)  # Add a line at zero
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.close()  # Close the plot to free up memory



def main():
    get_g3_histogram('por', 'GP')
    get_g3_histogram('por', 'MS')
    get_g3_histogram('mat', 'GP')
    get_g3_histogram('mat', 'MS')

    get_absences_plots('por')
    get_absences_plots('mat')
    make_poly_features('por')
    plot_feature_importance("../results/por/standard_scaling_poly_lasso_extended_train_0.1000.csv",
                            '', '../plots/por/feature_importance.png')
    plot_feature_importance("../results/mat/standard_scaling_poly_lasso_extended_train_0.2848.csv",
                            '', '../plots/mat/feature_importance.png')



if __name__ == '__main__':
    main()


