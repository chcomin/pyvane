from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile
from pyvesto import file_util, img_io, segmentation, skeleton, util, measure
from pyvesto.graph.creation import create_graph
import pyvesto.graph.adjustment as net_adjust

class Processor:

    def __init__(self, batch_name, in_folder, out_folder, out_folder_res, result_format, name_filter, run_steps, save_steps,
                 step_directories, channel_to_use, threshold, sigma, radius, comp_size, length_threshold,
                 tortuosity_scale, roi, roi_analysis_func, start_at, num_threads, verbosity):
        """Class for segmenting, representing and characterizing 2D or 3D blood vessel images. The
        methods contained here are very specific to some analyses done by Prof. Cesar Comin. They should
        be adapted to process blood vessel images for other experiments.

        Parameters
        ----------
        batch_name : str
            Name of the batch
        in_folder : Path or str
            Folder containing input images
        out_folder : Path or str
            Folder to save images generated from intermediate steps
        out_folder_res : Path or str
             Folder to save the results
        result_format : {'dict', 'table', 'disk'}
            How to save the results. If 'disk', a csv file is saved on the disk.
        name_filter : func
            Function for filtering the files. Returns True if the file should be processd and False otherwise.
        run_steps : dict
            Dictionary containing the processing steps to run. Has the form {'segmentation':{True,False},
            'skeletonization':{True,False}, 'network':{True,False}, 'analysis':{True,False}}
        save_steps : dict
            Dictionary containing which intermediary data should be saved on disk. Has the form
            {'segmentation':{True,False}, 'skeletonization':{True,False}, 'network-original':{True,False},
             'network-simple':{True,False}, 'network-final':{True,False}}
        step_directories : dict
            Directory names to save intermediary data. Has the form {'segmentation':'folder_name',
            'skeletonization':'folder_name', 'network-original':'folder_name',
            'network-simple':'folder_name', 'network-final':'folder_name'}
        channel_to_use : int
            Image channel to use when reading files.
        threshold : str
            Name of the file containing thresholds for segmentation. Can also be a float, in which case
            the value is used for all images.
        sigma : tuple of float
            Standard deviation of the Gaussian used for smoothing the image before segmentation
        radius : int
            Radius used for adaptive thresholding.
        comp_size : int
            Connected components smaller than `comp_size` will be removed after segmenting the image.
        length_threshold : float
            Threshold for removing small blood vessel branches.
        tortuosity_scale : float
            Scale used for calculating the tortuosity.
        start_at : int, optional
            Given the list of images to process, start at this index. If 0 or None, all images are processed.
        roi : tuple of slice, optional
            If not None, crop image before processing using the slices provided.
        num_threads : int
            Number of threads to use for skeleton calculation
        verbosity : {0, 1, 2, 3}
            Level of verbosity.
        storage : dict
            Dictionary for storing the calculated graphs and morphometry.
        """

        self.batch_name = batch_name
        self.in_folder = Path(in_folder)
        self.out_folder = Path(out_folder)
        self.out_folder_res = Path(out_folder_res)
        self.result_format = result_format
        self.name_filter = name_filter
        self.run_steps = run_steps
        self.save_steps = save_steps
        self.step_directories = step_directories
        self.channel_to_use = channel_to_use
        self.threshold = threshold
        self.sigma = sigma
        self.radius = radius
        self.comp_size = comp_size
        self.length_threshold = length_threshold
        self.tortuosity_scale = tortuosity_scale
        self.start_at = start_at
        self.roi = roi
        self.roi_analysis_func = roi_analysis_func
        self.num_threads = num_threads
        self.verbosity = verbosity
        self.storage = {}

    def run(self):
        """Start processing the images."""

        results = None
        run_steps = self.run_steps

        if run_steps['segmentation'] or run_steps['skeletonization'] or run_steps['network']:
            self.imgs_to_graphs(self.batch_name, self.in_folder, self.out_folder, self.name_filter, self.run_steps,
                                self.save_steps, self.step_directories, self.channel_to_use, self.threshold,
                                self.sigma, self.radius, self.comp_size, self.length_threshold, self.roi, self.start_at,
                                self.num_threads, self.verbosity)

        if run_steps['analysis']:
            in_folder_res = Path(self.out_folder)/'network/final'
            name_filter = lambda file: ('.' not in file) or ('pickle' in file)
            results = self.analyze_graphs(self.batch_name, in_folder_res, self.out_folder_res,
                                          self.tortuosity_scale, self.result_format, name_filter, self.verbosity,
                                          self.roi_analysis_func)

        return results

    def img_to_graph(self, file, in_folder, out_folder, run_steps, save_steps, step_directories, channel_to_use, threshold,
                     sigma, radius, comp_size, length_threshold, roi, num_threads, verbosity):
        """Convert a 2D or 3D blood vessel image into a graph."""

        filename = file.stem
        img = self.read_and_prepare_img(file, channel_to_use, roi)
        if save_steps['original']:
            img_proj = np.max(img.data, 0) if img.data.ndim==3 else img.data
            self.save_object_proj(file.parent, filename, img, False, img_proj)

        if run_steps['segmentation']:
            if verbosity>=2:
                print('Segmenting...')
            img_bin = segmentation.vessel_segmentation(img, threshold, sigma=sigma, radius=radius, comp_size=comp_size)
            if save_steps['segmentation']:
                out_sample_folder = self.gen_out_sample_folder(in_folder, out_folder, file, step_directories['segmentation'])
                img_bin_proj = np.sum(img_bin.data, 0) if img_bin.data.ndim==3 else img_bin.data
                self.save_object_proj(out_sample_folder, filename, img_bin, True, img_bin_proj, 'jet')

        if run_steps['skeletonization']:
            if verbosity>=2:
                print('Creating skeleton...')
            img_skel = skeleton.skeletonize(img_bin, num_threads=num_threads, verbosity=verbosity)
            if save_steps['skeletonization']:
                out_sample_folder = self.gen_out_sample_folder(in_folder, out_folder, file, step_directories['skeletonization'])
                img_skel_proj = np.max(img_skel.data, 0) if img_skel.data.ndim==3 else img_skel.data
                self.save_object_proj(out_sample_folder, filename, img_skel, True, img_skel_proj)

        if run_steps['network']:
            if verbosity>=2:
                print('Creating graph...')

            graph = create_graph(img_skel, verbose=(verbosity>=3))
            graph_simple = net_adjust.simplify(graph, False, verbose=(verbosity>=3))
            graph_final = net_adjust.adjust_graph(graph_simple, length_threshold, False, True, verbose=(verbosity>=3))

            if save_steps['network-original']:
                out_sample_folder = self.gen_out_sample_folder(in_folder, out_folder, file, step_directories['network-original'])
                self.save_graph(out_sample_folder, filename, graph, True)

            if save_steps['network-simple']:
                out_sample_folder = self.gen_out_sample_folder(in_folder, out_folder, file, step_directories['network-simple'])
                self.save_graph(out_sample_folder, filename, graph_simple, True)

            if save_steps['network-final']:
                out_sample_folder = self.gen_out_sample_folder(in_folder, out_folder, file, step_directories['network-final'])
                self.save_graph(out_sample_folder, filename, graph_final, True)

                self.storage['graph_final'] = graph_final

        return graph_final

    def imgs_to_graphs(self, batch_name, in_folder, out_folder, name_filter, run_steps, save_steps, step_directories,
                      channel_to_use, threshold, sigma, radius, comp_size, length_threshold, roi, start_at,
                      num_threads, verbosity):
        """Convert a set of 2D or 3D blood vessel images into graphs."""

        out_folder = Path(out_folder)
        files = self.prepare_files(in_folder, batch_name, out_folder, save_steps, step_directories, name_filter)
        num_files = len(files)
        if (start_at is None) or (start_at==0):
            first_index = 0
        else:
            first_index = None
            for idx, file in enumerate(files):
                if start_at in str(file):
                    first_index = idx
                    break
            if first_index is None:
                print('Warning, tried to start at file {} but the file was not found. Starting from the beginning')
                first_index = 0
            files = files[first_index:]

        #import pdb; pdb.set_trace()
        if isinstance(threshold, str):
            threshold_dict = self.load_thresholds(threshold)

        for idx, file in enumerate(files):

            filename = file.stem
            if verbosity:
                print(f'Processing file {filename} ({first_index+idx+1} of {num_files})...')

            if isinstance(threshold, str):
                file_tag = self.get_file_tag(file)
                threshold = threshold_dict[file_tag]
            self.img_to_graph(file, in_folder, out_folder, run_steps, save_steps,
                              step_directories, channel_to_use, threshold, sigma, radius,
                              comp_size, length_threshold, roi, num_threads, verbosity)

    def analyze_graph(self, file, tortuosity_scale, img_roi=None):
        """Calculate the length, number of branching points and tortuosity of a blood vessel graph."""

        print(file)
        graph_final = pickle.load(open(file, 'rb'))
        length = measure.vessel_density(graph_final, graph_final.graph['shape'], img_roi=img_roi, scale_factor=1e-3)
        num_branches = measure.branch_point_density(graph_final, graph_final.graph['shape'], img_roi=img_roi,
                                                    scale_factor=1e-3)
        tortuosity = measure.tortuosity(graph_final, tortuosity_scale, True)

        return length, num_branches, tortuosity

    def analyze_graphs(self, batch_name, in_folder_res, out_folder_res, tortuosity_scale, result_format,
                       name_filter, verbosity, roi_analysis_func):
        """Calculate the length, number of branching points and tortuosity of a set of
        blood vessel graphs."""

        in_folder_batch = Path(in_folder_res)/batch_name
        root = Path(in_folder_batch)
        _, files_net = file_util.get_files(root, name_filter)

        if verbosity:
            num_files = len(files_net)
            print('Calculating properties...')

        img_roi = None
        results = {}
        for idx, file in enumerate(files_net):

            filename = file.stem
            file_tag = self.get_file_tag(file)
            if verbosity:
                print(f'Processing file {filename} ({idx+1} of {num_files})...', end='\r')

            if roi_analysis_func is not None:
                img_roi = roi_analysis_func(file)

            length, num_branches, tortuosity = self.analyze_graph(file, tortuosity_scale, img_roi=img_roi)
            results[file_tag] = {'length':length, 'num_branches':num_branches, 'tortuosity':tortuosity}
        print('')

        self.storage['results'] = results

        if result_format=='dict':
            pass
        elif result_format=='table' or result_format=='disk':
            results = self.generate_table(results)
            if result_format=='disk':
                if not out_folder_res.is_dir():
                    os.mkdir(out_folder_res)
                open(f'{out_folder_res/batch_name}.tsv', 'w').write(results)

        if verbosity:
            print('Done!')

        return results

    def generate_table(self, results):
        """Generate nice table containing blood vessel morphometry."""

        header = "Name\tLength (mm/mm^3)\tBranching points (1/mm^3)\tTortuosity"

        table_str = header+'\n'
        for filename in results:
            graph_meas = results[filename]
            length, num_branches, tortuosity = graph_meas['length'], graph_meas['num_branches'], graph_meas['tortuosity']
            line_str = f'{filename}\t{length}\t{num_branches}\t{tortuosity}\n'
            table_str += line_str

        return table_str

    def prepare_files(self, in_folder, batch_name, out_folder, save_steps, step_directories, name_filter):
        """Obtain list of files for processing and generate necessary folders for saving intermediary
        steps of the pipeline (segmented blood vessels, skeleton, etc)."""

        # Read file list and make output folders
        in_folder_batch = Path(in_folder)/batch_name
        root = Path(in_folder_batch)
        file_tree, files = file_util.get_files(root, name_filter)
        step_dirs = [step_directories[step] for step, save in save_steps.items() if save]
        file_util.make_directories(file_tree, out_folder, step_dirs, ['maximum_projection', 'pickle'])

        return files

    def gen_out_sample_folder(self, in_folder, out_folder, file, step):
        """Generate a new folder path."""

        file_rel_path = file.relative_to(in_folder)
        return (out_folder/step/file_rel_path).parent

    def read_and_prepare_img(self, file, channel=0, roi=None):
        """Read image form disk, transform its data to float, apply the linear transformation
        [min, max]->[0, 255] and interpolate the image to make it isotropic."""

        img = img_io.read_img(file, channel=channel)
        if roi is not None:
            img.data = img.data[roi]
            img.shape = img.data.shape
        img_dtype = img.data.dtype
        if img_dtype!=np.uint16 and img_dtype!=np.uint8:
            raise ValueError(f'Pixel data type is {img_dtype}, but should be either uint8 or uint16.')
        img.to_float()
        img.stretch_data_range(255)
        img.make_isotropic()

        return img

    def save_object_proj(self, directory, name, obj, save_obj=True, obj_proj=None, cmap=None):
        """Save an object as a pickle file. Optionally, also save an image representing the object.
        This is useful for saving a 3D image together with a 2D projection or a graph and its
        respective image."""

        if cmap is None:
            cmap = 'gray'

        directory = Path(directory)

        if save_obj:
            out_pickle_dir = directory/'pickle'
            if not out_pickle_dir.is_dir():
                os.mkdir(out_pickle_dir)
            pickle.dump(obj, open(out_pickle_dir/(name+'.pickle'), 'wb'))

        if obj_proj is not None:
            out_proj_dir = directory/'maximum_projection'
            if not out_proj_dir.is_dir():
                os.mkdir(out_proj_dir)
            plt.imsave(out_proj_dir/(name+'.png'), obj_proj, cmap=cmap)

    def save_graph(self, directory, name, graph, save_graph=True, cmap=None, node_pixels_color=(0, 0, 0)):
        """Save a graph to disk."""

        directory = Path(directory)

        img_graph = util.graph_to_img(graph, node_color=(255, 255, 255), node_pixels_color=node_pixels_color,
                                      edge_color=(255, 255, 255))
        img_graph_proj = np.max(img_graph, 0) if img_graph.ndim==4 else img_graph   # Graph image has color
        self.save_object_proj(directory, name, graph, save_obj=save_graph, obj_proj=img_graph_proj, cmap=cmap)

    def load_thresholds(self, filename):
        """Load file containing the thresholds for blood vessel segmentation."""

        data = open(filename, 'r').readlines()

        threshold_dict = {}
        in_batch = False
        for line_index, line in enumerate(data):
            if line[0]=='#':
                in_batch = False
                batch_name = line[1:].strip()
                if batch_name==self.batch_name:
                    in_batch = True
            elif in_batch:
                splitted_line = line.strip().split('\t')
                index, stack_name, best_threshold = splitted_line
                threshold_dict[stack_name] = float(best_threshold)

        return threshold_dict

    def try_thresholds(self, threshold_values, out_folder):

        batch_name = self.batch_name
        in_folder = self.in_folder
        name_filter = self.name_filter
        out_folder = Path(out_folder)/batch_name

        in_folder_batch = Path(in_folder)/batch_name
        root = Path(in_folder_batch)
        file_tree, files = file_util.get_files(root, name_filter)
        num_files = len(files)

        for idx, file in enumerate(files):

            filename = file.stem
            if self.verbosity:
                print(f'Processing file {filename} ({idx+1} of {num_files})...')

            img = self.read_and_prepare_img(file, self.channel_to_use)

            for threshold in threshold_values:
                img_bin = segmentation.vessel_segmentation(img, threshold, sigma=self.sigma, radius=self.radius,
                                                comp_size=self.comp_size)
                img_bin_all_comps = segmentation.vessel_segmentation(img, threshold, sigma=self.sigma, radius=self.radius,
                                                comp_size=0)

                img_bin_np = img_bin.data
                img_diff = np.logical_xor(img_bin_np, img_bin_all_comps.data)
                img_final_diff = img_bin_np.astype(np.uint8)*2
                img_final_diff[img_diff] = 1		

                if not out_folder.is_dir():
                    os.mkdir(out_folder)
                if img_final_diff.ndim==3:
                    img_out_proj = np.max(img_final_diff, axis=0)
                else:
                    img_proj_out = img_final_diff
                file_tag = self.get_file_tag(file)
                plt.imsave(out_folder/f'{file_tag}_{threshold:.1f}.png', img_proj_out, cmap='hot')

    def print_batch_files(self, only_files=True, replace_slash=None):

        if replace_slash is None:
            replace_slash = '/'

        in_folder_batch = Path(self.in_folder)/self.batch_name
        root = Path(in_folder_batch)
        file_tree, files = file_util.get_files(root, self.name_filter)
        for file in files:
            if only_files:
                name_to_print = file.stem
            else:
                name_to_print = self.get_file_tag(file, sep=replace_slash)
            print(name_to_print)

    def get_file_tag(self, file, sep='@'):

            file_parts = file.parts
            ind = file_parts.index(self.batch_name)
            tag = sep.join(file_parts[ind+1:-1])
            tag += '@'+file.stem

            return tag

    def verify_results(self, out_file):

        in_folder_batch = Path(self.in_folder)/self.batch_name
        root = Path(in_folder_batch)
        file_tree, files = file_util.get_files(root, self.name_filter)
        out_folder = Path(self.out_folder)
        num_files = 2*len(files)


        for idx, file in enumerate(files):
            filename = file.stem
            img = self.read_and_prepare_img(file, self.channel_to_use, self.roi)

            out_sample_folder = self.gen_out_sample_folder(self.in_folder, out_folder, file, self.step_directories['network-final'])
            out_sample_folder = out_sample_folder/'pickle'/f'{filename}.pickle'
            graph = pickle.load(open(out_sample_folder, 'rb'))
            img_graph = util.graph_to_img(graph, node_color=(255, 255, 255), node_pixels_color=(0, 0, 0),
                                      edge_color=(255, 255, 255))
            img_graph = img_graph[:,:,0]

            if img.ndim>=3:
                img_proj = np.max(img.data, axis=0)
            else:
                img_proj = img.data

            if idx==0:
                verification_stack = np.zeros((num_files, img_proj.shape[0], img_proj.shape[1]), dtype=np.uint8)

            verification_stack[2*idx] = np.round(img.data).astype(int)
            verification_stack[2*idx+1] = img_graph


        tifffile.imsave(out_file, verification_stack)