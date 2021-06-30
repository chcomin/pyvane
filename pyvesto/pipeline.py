from pathlib import Path
import copy
import os
import pickle
from functools import partial
from collections import defaultdict
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile

from . import file_util, img_io, segmentation, skeleton, util, measure
from .graph.creation import create_graph
from .graph import adjustment as net_adjust

class BasePipeline:

    DEFAULT_STEPS = ('segmentation', 'skeletonization', 'network', 'analysis')
    DIRECTORY_NAMES = ('binary', 'skeleton', 'network', 'results')

    def __init__(self, input_path, img_reader=None, batch_name=None, output_path='./', result_format='disk', name_filter=None, save_steps='all',
                 start_at=0, verbosity=0):
        """Class for segmenting, representing and characterizing 2D or 3D blood vessel images. The
            methods contained here are very specific to some analyses done by Prof. Cesar Comin. They should
            be adapted to process blood vessel images for other experiments.

            Parameters
            ----------
            input_path : Path or str
                Folder containing input images
            output_path : Path or str
                Folder to save images generated from intermediate steps
            channel_to_use : int
                Image channel to use when reading files.
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
            start_at : int or str, optional
                Given the list of images to process, start at this index. If 0 or None, all images are processed.
            roi_process_func : tuple of slice, optional
                If not None, crop image before processing using the slices provided.
            roi_analysis_func : tuple of slice, optional
                If not None, crop image before processing using the slices provided.
            num_threads : int
                Number of threads to use for skeleton calculation
            verbosity : {0, 1, 2, 3}
                Level of verbosity.
            """

        input_path = Path(input_path)
        output_path = Path(output_path)

        if batch_name is None:
            batch_name = input_path.parts[-1]
        if batch_name != output_path.parts[-1]:
            output_path /= batch_name
        if name_filter is None:
            name_filter = lambda x: True
        elif isinstance(name_filter, (list, tuple, set)):
            # If `name_filter` is a set allowing `x in name_filter`, turn it into `name_filter(x)``
            name_filter = (lambda name_set: (lambda file: file in name_set))(name_filter) 
        if save_steps=='all':
            save_steps = self.DEFAULT_STEPS[:-1]

        self.input_path = input_path
        self.batch_name = batch_name
        self.output_path = output_path
        self.img_reader = img_reader
        self.result_format = result_format
        self.name_filter = name_filter
        self.save_steps = save_steps
        self.start_at = start_at
        self.verbosity = verbosity

        files = self.prepare_files()
        self.files = files
        self.storage = {}
        self.run_steps = []

    def set_processors(self, segmenter=None, skeleton_builder=None, network_builder=None, analyzer=None):

        self.segmenter = segmenter
        self.skeleton_builder = skeleton_builder
        self.network_builder = network_builder
        self.analyzer = analyzer
        self._generate_steps()
        self.create_folders()

    def prepare_files(self):
        """Obtain list of files for processing and generate necessary folders for saving intermediary
        steps of the pipeline (segmented blood vessels, skeleton, etc)."""

        # Read file list and make output folders
        file_tree, files = file_util.get_files(self.input_path, self.name_filter)
        if 'maximum_projection' in files:
            files.remove('maximum_projection')

        start_at = self.start_at
        if isinstance(start_at, int):
            first_index = start_at
        else:
            first_index = None
            for idx, file in enumerate(files):
                if start_at in str(file.name):
                    first_index = idx
                    break
            if first_index is None:
                raise ValueError('Warning, tried to start at file {start_at} but the file was not found.')
        files = files[first_index:]

        return files

    def _generate_steps(self):

        run_steps = []
        if self.segmenter is not None:
            run_steps.append('segmentation')
        if self.skeleton_builder is not None:
            run_steps.append('skeletonization')
        if self.network_builder is not None:
            run_steps.append('network')
        if self.analyzer is not None:
            run_steps.append('analysis')
        self.run_steps = run_steps

    def create_folders(self):

        output_path = self.output_path
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for step in self.save_steps:
            step_idx = self.DEFAULT_STEPS.index(step)
            directory = self.DIRECTORY_NAMES[step_idx]
            output_path_step = output_path/directory
            if not os.path.isdir(output_path_step):
                os.mkdir(output_path_step)

        output_path_res = output_path/'results'
        if not os.path.isdir(output_path_res):
            os.mkdir(output_path_res)

    def run(self):

        result_format = self.result_format
        if result_format=='disk':
            output_file = f'{self.output_path}/{self.DIRECTORY_NAMES[-1]}/{self.batch_name}.tsv'

        num_files = len(self.files)
        self.results = {}
        for idx, file in enumerate(self.files):
            filename = file.stem
            if self.verbosity:
                print(f'Processing file {filename} ({idx+1} of {num_files})...')

            measurements = self._run_one_file(file)
            self.results[filename] = measurements

            if measurements is not None and result_format=='disk':
                if idx==0:
                    with open(output_file, 'w') as output_fd:
                        header = self.generate_header(measurements.keys())
                        output_fd.write(header)
                with open(output_file, 'a') as output_fd:
                    line_str = self.generate_line(filename, measurements)
                    output_fd.write(line_str)

        return self.results

    def _run_one_file(self, file):

        filename = file.stem

        if 'segmentation' in self.run_steps:
            img = self.img_reader(file)
            is_3d = img.ndim==3
            if is_3d:
                img_proj = np.max(img.data, 0)
                output_dir = self.output_path/'original'
                self.save_object_proj(output_dir, filename, img, False, img_proj)

            img = self.segmenter(img, file)
            if 'segmentation' in self.save_steps:
                output_dir = self.get_output_directory('segmentation')
                img_proj = np.max(img.data, axis=0) if is_3d else img.data
                self.save_object_proj(output_dir, filename, img, True, img_proj, 'jet')
        else:
            input_dir = self.get_output_directory('segmentation')
            try:
                img = pickle.load(open(input_dir/f'{filename}.pickle', 'rb'))
            except FileNotFoundError:
                raise FileNotFoundError("No processor for segmentation step and no binary image found.")

        if 'skeletonization' in self.run_steps:
            img = self.skeleton_builder(img, file)
            is_3d = img.ndim==3
            if 'skeletonization' in self.save_steps:
                output_dir = self.get_output_directory('skeletonization')
                img_proj = np.max(img.data, axis=0) if is_3d else img.data
                self.save_object_proj(output_dir, filename, img, True, img_proj, 'gray')
        elif 'network' in self.run_steps:
            input_dir = self.get_output_directory('skeletonization')
            try:
                img = pickle.load(open(input_dir/f'{filename}.pickle', 'rb'))
            except FileNotFoundError:
                raise FileNotFoundError("No processor for skeletonization step and no skeleton image found.")

        if 'network' in self.run_steps:
            network = self.network_builder(img, file)
            if 'network' in self.save_steps:
                output_dir = self.get_output_directory('network')
                self.save_graph(output_dir, filename, network, True)
        elif 'analysis' in self.run_steps:
            input_dir = self.get_output_directory('network')
            try:
                network = pickle.load(open(input_dir/f'{filename}.pickle', 'rb'))
            except FileNotFoundError:
                raise FileNotFoundError("No processor for network step and no graph file found.")

        measurements = None
        if 'analysis' in self.run_steps:
            measurements = self.analyzer(network, file)

        return measurements

    def get_output_directory(self, step):

        step_idx = self.DEFAULT_STEPS.index(step)
        directory = self.DIRECTORY_NAMES[step_idx]

        return self.output_path/directory

    def generate_header(self, column_names):

        header = 'Name'
        for name in column_names:
            header += f"\t{name}"
        header = header+'\n'  

        return header      

    def generate_line(self, filename, measurements):

        line_str = f'{filename}'
        for key, value in measurements.items():
            line_str += f'\t{value}'
        line_str += '\n'

        return line_str

    def generate_table(self, results):
        """Generate nice table containing blood vessel morphometry."""

        header = self.generate_header(results[first_filename].keys())
        table_str = header
        for filename, measurements in results.items():
            line_str = self.generate_line(filename, measurements)
            table_str += line_str

        return table_str

    def save_object_proj(self, directory, name, obj, save_obj=True, obj_proj=None, cmap=None):
        """Save an object as a pickle file. Optionally, also save an image representing the object.
        This is useful for saving a 3D image together with a 2D projection or a graph and its
        respective image."""

        if cmap is None:
            cmap = 'gray'

        directory = Path(directory)

        if save_obj:
            if not directory.is_dir():
                os.mkdir(directory)
            pickle.dump(obj, open(directory/(name+'.pickle'), 'wb'))

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

    def print_batch_files(self, only_stem=True, replace_slash=None):

        if replace_slash is None:
            replace_slash = '/'

        for file in self.files:
            if only_stem:
                name_to_print = file.stem
            else:
                name_to_print = self.get_file_tag(file, sep=replace_slash)
            print(name_to_print)

    def get_file_tag(self, file, sep='@'):

        return file_util.get_file_tag(file, self.input_path.stem)

class BaseProcessor:

    def __init__(self):
        pass

    def __call__(self, img, file=None):
        return self.apply(img, file)

    def apply(self, img, file=None):
        pass

class DefaultSegmenter(BaseProcessor):

    def __init__(self, threshold, sigma, radius=40, comp_size=500, hole_size=None, batch_name=None):

        if isinstance(threshold, str):
            threshold = self.load_thresholds(threshold, batch_name)
        elif isinstance(threshold, dict):
            threshold = threshold
        else:
            # Fixed threshold for all images
            threshold = defaultdict((lambda threshold: (lambda:threshold))(threshold))

        self.threshold = threshold
        self.sigma = sigma
        self.radius = radius
        self.comp_size = comp_size
        self.hole_size = hole_size
        self.batch_name = batch_name

    def apply(self, img, file=None):

        filename = file.stem
        threshold = self.threshold[filename]

        return segmentation.vessel_segmentation(img, threshold, sigma=self.sigma, radius=self.radius, comp_size=self.comp_size, 
                        hole_size=self.hole_size)

    def load_thresholds(self, filename, batch_name):
        """Load file containing the thresholds for blood vessel segmentation."""

        data = open(filename, 'r').readlines()

        threshold_dict = {}
        in_batch = False
        for line_index, line in enumerate(data):
            if line[0]=='#':
                in_batch = False
                batch_name = line[1:].strip()
                if batch_name==batch_name:
                    in_batch = True
            elif in_batch:
                splitted_line = line.strip().split('\t')
                index, stack_name, best_threshold = splitted_line
                threshold_dict[stack_name] = float(best_threshold)

        return threshold_dict

class DefaultSkeletonBuilder(BaseProcessor):

    def __init__(self, num_threads=1, verbosity=0):

        self.num_threads = num_threads
        self.verbosity = verbosity

    def apply(self, img, file=None):

        return skeleton.skeletonize(img, num_threads=self.num_threads, verbosity=self.verbosity)

class DefaultNetworkBuilder(BaseProcessor):

    def __init__(self, length_threshold=10, verbosity=0):

        self.length_threshold = length_threshold
        self.verbosity = verbosity

    def apply(self, img, file=None):

        graph = create_graph(img, verbose=(self.verbosity>=3))
        graph_simple = net_adjust.simplify(graph, False, verbose=(self.verbosity>=3))
        graph_final = net_adjust.adjust_graph(graph_simple, self.length_threshold, False, True, verbose=(self.verbosity>=3))

        return graph_final

class DefaultAnalyzer(BaseProcessor):

    def __init__(self, tortuosity_scale):

        self.tortuosity_scale = tortuosity_scale

    def apply(self, graph, file=None):

        img_roi = self.load_roi(file)
        length = measure.vessel_density(graph, graph.graph['shape'], img_roi=img_roi, scale_factor=1e-3)
        num_branches = measure.branch_point_density(graph, graph.graph['shape'], img_roi=img_roi,
                                                    scale_factor=1e-3)
        tortuosity = measure.tortuosity(graph, self.tortuosity_scale, True)     

        measurements = {'Length (mm/mm^3)':length, 'Branching points (1/mm^3)':num_branches, 'Tortuosity':tortuosity}  

        return measurements

    def load_roi(self, file):

        return None
        
class AuxiliaryPipeline(BasePipeline):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def set_processors(self, segmenter):

        self.segmenter = segmenter
        self.create_folders()

    def create_folders(self):

        output_path = self.output_path
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        output_path_thresholded = output_path/'threshold_tests'
        if not os.path.isdir(output_path_thresholded):
            os.mkdir(output_path_thresholded)

    def try_thresholds(self, threshold_values):

        files = self.files
        output_path = self.output_path/'threshold_tests'
        
        num_files = len(files)
        for idx, file in enumerate(files):

            filename = file.stem
            if self.verbosity:
                print(f'Processing file {filename} ({idx+1} of {num_files})...')

            img = self.img_reader(file)
            for threshold in threshold_values:
                self.segmenter.threshold = defaultdict((lambda threshold: (lambda:threshold))(threshold))
                ref_segmenter = copy.deepcopy(self.segmenter)
                ref_segmenter.comp_size = 0

                img_bin = self.segmenter(img, file)
                img_bin_all_comps = ref_segmenter(img, file)

                img_bin_np = img_bin.data
                img_diff = np.logical_xor(img_bin_np, img_bin_all_comps.data)
                img_final_diff = img_bin_np.astype(np.uint8)*2
                img_final_diff[img_diff] = 1		

                if img_final_diff.ndim==3:
                    img_proj_out = np.max(img_final_diff, axis=0)
                else:
                    img_proj_out = img_final_diff

                plt.imsave(output_path/f'{filename}_{threshold:.1f}.png', img_proj_out, cmap='hot')

    def verify_results(self):

        files = self.files
        num_files = 2*len(files)

        for idx, file in enumerate(files):

            filename = file.stem
            img = self.img_reader(file)

            output_graph_img = self.output_path/self.DIRECTORY_NAMES[2]/'maximum_projection'/f'{filename}.png'
            img_graph = plt.imread(output_graph_img)
            #img_graph = pickle.load(open(output_graph_img, 'rb'))
            img_graph = 255*img_graph[:,:,0]

            if img.ndim>=3:
                img_proj = np.max(img.data, axis=0)
            else:
                img_proj = img.data

            if idx==0:
                verification_stack = np.zeros((num_files, img_proj.shape[0], img_proj.shape[1]), dtype=np.uint8)

            verification_stack[2*idx] = np.round(img_proj).astype(int)
            verification_stack[2*idx+1] = np.round(img_graph).astype(int)


        tifffile.imsave(self.output_path/'verification.tif', verification_stack)

def read_and_adjust_img(file, channel=0, roi=None):
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

def build_default_pipeline(input_path, output_path='./', channel=None, name_filter=None, only_analyze=False,
                           start_at=0, verbosity=0):

    img_reader = partial(read_and_adjust_img, channel=channel)

    bp = BasePipeline(input_path, img_reader, output_path=output_path, name_filter=name_filter, start_at=start_at, verbosity=verbosity)

    segmenter = DefaultSegmenter(0, None)
    skeleton_builder = DefaultSkeletonBuilder()
    network_builder = DefaultNetworkBuilder()
    analyzer = DefaultAnalyzer(5)

    bp.set_processors(segmenter, skeleton_builder, network_builder, analyzer)

    return bp

    