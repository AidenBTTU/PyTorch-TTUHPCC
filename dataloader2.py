import os
import tarfile
import numpy as np
import awkward as ak
import uproot
import vector

def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_nparticles', 'jet_sdmass', 'jet_tau2', 'jet_tau3', 'jet_tau4', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'],
        use_extracted=False):
    """Loads all ROOT files from a `.tar` archive or a directory containing extracted ROOT files."""

    def _extract_tar(tar_path, extract_dir):
    """Extracts a .tar file and organizes files into subdirectories based on their base prefix."""
        if not os.path.exists(extract_dir):
            print(f"Extracting {tar_path} to {extract_dir}...")
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
        else:
            print(f"Using existing extracted folder: {extract_dir}")

    # Organize the ROOT files into subdirectories based on their base prefix
        root_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.root'):
                # Get the base prefix (before any numbers or underscores)
                    prefix = ''.join([char for char in file.split('.')[0] if not char.isdigit() and char != '_'])

                # Create a subdirectory for each base prefix
                    subdir = os.path.join(extract_dir, prefix)
                    os.makedirs(subdir, exist_ok=True)

                # Move the file to the appropriate subdirectory
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(subdir, file)
                    os.rename(old_file_path, new_file_path)

                # Add the file to the list of root files
                    root_files.append(new_file_path)

        if not root_files:
            raise FileNotFoundError(f"No ROOT files found in {extract_dir}")
        return root_files


    def _pad(a, maxlen, value=0, dtype='float32'):
        """Pads or truncates arrays to maxlen."""
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    root_files = []
    if filepath.endswith('.tar'):
        extract_dir = filepath.replace('.tar', '')  # Create folder name from tar file
        if use_extracted and os.path.exists(extract_dir):  # Use existing folder
            print(f"Using already extracted folder: {extract_dir}")
            root_files = _extract_tar(filepath, extract_dir)
        else:  # Extract the tar file
            root_files = _extract_tar(filepath, extract_dir)
    elif os.path.isdir(filepath):  # Directly use an extracted folder
        print(f"Using extracted folder: {filepath}")
        root_files = _extract_tar(None, filepath)
    else:
        root_files = [filepath]  # Single ROOT file

    all_x_particles, all_x_jets, all_y = [], [], []

    for root_file in root_files:
        print(f"Processing: {root_file}")
        table = uproot.open(root_file)['tree'].arrays()

        # Compute derived features
        p4 = vector.zip({'px': table['part_px'],
                         'py': table['part_py'],
                         'pz': table['part_pz'],
                         'energy': table['part_energy']})
        table['part_pt'] = p4.pt
        table['part_eta'] = p4.eta
        table['part_phi'] = p4.phi

        # Convert to NumPy arrays
        x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
        x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
        y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

        # Append data from each ROOT file
        all_x_particles.append(x_particles)
        all_x_jets.append(x_jets)
        all_y.append(y)

    # Concatenate all loaded data
    all_x_particles = np.concatenate(all_x_particles, axis=0)
    all_x_jets = np.concatenate(all_x_jets, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_x_particles, all_x_jets, all_y

