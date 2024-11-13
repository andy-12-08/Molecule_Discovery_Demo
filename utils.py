import numpy as np
import selfies as sf
from rdkit import Chem
import torch
import sys
sys.path.append('../')
from src.vae_module import flatten
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA


# Function to encode the input data from SMILES to one-hot encoded SELFIES
def encode_data_from_smiles_to_vae_encoded(input_smiles_list):
    input_selfies_list = list(map(sf.encoder, input_smiles_list))
    # Parameters for encoding
    max_len = max(sf.len_selfies(s) for s in input_selfies_list)
    alphabet = sf.get_alphabet_from_selfies(input_selfies_list)
    alphabet.add("[nop]")
    alphabet = list(sorted(alphabet))
    vocab_stoi = {symbol: idx for idx, symbol in enumerate(alphabet)}
    vocab_itos = {idx: symbol for symbol, idx in vocab_stoi.items()}
    # Convert SELFIES to one-hot encoding
    input_one_hot_arr = np.array([sf.selfies_to_encoding(s, vocab_stoi, pad_to_len=max_len)[1] for s in input_selfies_list])
    ## Convert the "input_one_hot_arr" dataset to tensor
    input_one_hot_arr_tensor = torch.tensor(input_one_hot_arr, dtype=torch.float32)
    # Flatten the input data using the custom 'flatten' function
    # This function takes a 3D tensor 'x_train' and reshapes it into a 2D tensor
    width, height, input_dim, flattened_dataset = flatten(input_one_hot_arr_tensor)
    return height, vocab_itos, width

# Function to generate molecules from the trained VAE
def generate_vae_molecules(n, height, vocab_itos, width, vae, latent_dim):
    # Generate molecules from the trained VAE
    vae.eval()
    with torch.no_grad():
        latent_samples = torch.randn(n, latent_dim)
        generated_molecules = vae.decoder(latent_samples)
        # Convert the generated molecules to SMILES
        def generated_molecules_to_smiles(generated_molecules):
            # Reshape satisfying_molecules_tensor back to a 3D tensor
            generated_molecules_tensor_3d = generated_molecules.view(-1, width, height)
            # Convert the PyTorch 3D tensor to a NumPy array
            generated_molecules_numpy = generated_molecules_tensor_3d.numpy()
            max_values = np.max(generated_molecules_numpy, axis=2, keepdims=True)
            generated_data = np.where(generated_molecules_numpy == max_values, 1, 0)
            ### Reproduce SMILES list and visualize the output images
            output_smiles_list = []
            for i in range (0,len(generated_data)):
                sm = sf.decoder(sf.encoding_to_selfies(generated_data[i].tolist(), vocab_itos, enc_type="one_hot"))
                output_smiles_list.append(sm)
            return output_smiles_list
        # Convert the generated molecules to SMILES
        generated_molecules_smiles_list = generated_molecules_to_smiles(generated_molecules)
        # convert smiles to canonical smiles
        generated_molecules_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(sm), canonical=True) for sm in generated_molecules_smiles_list]
        # remove duplicates
        generated_molecules_smiles_list = list(set(generated_molecules_smiles_list))
        return generated_molecules_smiles_list
    
# Function to generate molecules from a latent vector using the VAE model
def generate_molecules_from_latent(latent_vector, height, vocab_itos, width, vae):
    with torch.no_grad():  # Disable gradient calculation for efficiency
        generated_molecules = vae.decoder(latent_vector)  # Decode the latent vector
        # Reshape the decoded output to a 3D tensor
        generated_molecules_tensor_3d = generated_molecules.view(-1, width, height)
        generated_molecules_numpy = generated_molecules_tensor_3d.numpy()
        # Convert the output to a one-hot encoded format
        max_values = np.max(generated_molecules_numpy, axis=2, keepdims=True)
        generated_data = np.where(generated_molecules_numpy == max_values, 1, 0)
        # Convert the one-hot encoded molecules back to SMILES strings
        output_smiles_list = []
        for data in generated_data:
            sm = sf.decoder(sf.encoding_to_selfies(data.tolist(), vocab_itos, enc_type="one_hot"))
            output_smiles_list.append(sm)
        # # Canonicalize the SMILES and remove duplicates
        # output_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(sm), canonical=True) for sm in output_smiles_list if Chem.MolFromSmiles(sm)]
        # output_smiles_list = list(set(output_smiles_list))  # Remove duplicates
        return output_smiles_list
    
# Function to produce a descriptor dataframe from a SMILES dataframe
def features_df(data_df):

    # Get a list of descriptor functions
    descriptor_list = [desc[0] for desc in Descriptors._descList]
    descriptor_list = descriptor_list[:100]  # Limit to first 100 descriptors for simplicity

    # drop ['NumRadicalElectrons'] descriptors
    descriptor_list = [desc for desc in descriptor_list if desc not in ['NumRadicalElectrons']]

    # Function to calculate descriptors
    def compute_descriptors(smiles, descriptor_functions):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * len(descriptor_functions)
        return [func(mol) if mol else None for func in descriptor_functions]

    # Map descriptor names to functions
    descriptors_functions = [getattr(Descriptors, name) for name in descriptor_list]

    # Compute descriptors for each molecule
    descriptors_data = [compute_descriptors(sm, descriptors_functions) for sm in data_df['smiles']]

    # Create DataFrame of descriptors
    descriptors_df = pd.DataFrame(descriptors_data, columns=descriptor_list)

    # Merge descriptors_df with the original data
    df = pd.concat([data_df, descriptors_df], axis=1)

    # what columns have all rows as zero 
    zero_cols = df.columns[(df == 0).all()]

    # remove columns with all zeros
    df = df.drop(zero_cols, axis=1)

    df.drop(columns=list(data_df.columns), inplace=True)

    # scale the data
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # fill NaN values with 0
    df.fillna(0, inplace=True)

    return df

# Function to generate images of the generated molecules in the latent space
def generate_images(original_data, mol, prop, episode, step, reward):
    # create a dataframe for the generated molecule
    generated_smiles_list_df = pd.DataFrame({'smiles': mol, 'logCMC': prop})
    # add the generated molecule to the original data
    merge_df = pd.concat([original_data, generated_smiles_list_df], axis=0)
    merge_df.reset_index(drop=True, inplace=True)
    pca = PCA(n_components=2)
    # fitter = pca.fit(features_df(merge_df.iloc[:-1, :]))
    # use the filter to transform the data
    # pca_df = fitter.transform(features_df(merge_df))
    pca_df = pca.fit_transform(features_df(merge_df))
    # Sample PCA data and logCMC values
    pca_x = pca_df[:, 0]
    pca_y = pca_df[:, 1]# 
    logCMC = merge_df['logCMC'].values

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize the color scale for the data
    norm = mcolors.Normalize(vmin=logCMC.min(), vmax=logCMC.max())
    cmap = plt.cm.RdYlBu

    # Use scatter directly if you want to avoid binning
    sc = ax.scatter(pca_x, pca_y, c=logCMC, cmap=cmap, norm=norm, s=50, alpha=0.8)

    # Add the generated molecule point
    generated_molecule_value = logCMC[-1]
    generated_molecule_color = cmap(norm(generated_molecule_value))

    if reward >= 0:
        reward_txt = 'reward'
    else:
        reward_txt = 'penalty'
    ax.scatter(
        pca_df[-1, 0], pca_df[-1, 1], 
        color=generated_molecule_color, 
        marker='P', s=250, edgecolor='black', linewidth=1.5,
        label=f'Generated Molecule \n episode={episode}, step={step} \n property={generated_molecule_value:.2f}, {reward_txt}={reward:.2f}'
    )
    # Set axis labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Agent Activity in Latent Space')
    # Add colorbar
    fig.colorbar(sc, ax=ax, label='Property value')
    # Add legend
    ax.legend(loc='upper left')
    # save the plot with high resolution
    plt.savefig(f'RL_Results/movie/episode_{episode}_step{step}.png', dpi=300)
    plt.close()


