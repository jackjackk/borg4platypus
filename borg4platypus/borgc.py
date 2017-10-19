from platypus.algorithms import Algorithm
from platypus.core import Solution, Problem
from tqdm import tqdm
import logging
import os
import sys
import numpy as np

LOGGER = logging.getLogger("borg4platypus")
DEFAULT_EXTRA_BORGLIBDIR = os.path.join(os.environ['HOME'], 'tools', 'borg', 'build', 'lib')


def _load_default_libpath():
    for libpath in [f'{prefix_libpath}_LIBRARY_PATH' for prefix_libpath in ['DYLD', 'LD']]:
        os.environ[libpath] = os.environ.get(libpath, '') + os.pathsep + DEFAULT_EXTRA_BORGLIBDIR
_load_default_libpath()
from borg import Direction


class BorgC(Algorithm):

    @staticmethod
    def load_default_libpath():
        _load_default_libpath()

    def __init__(self, problem, epsilons, seed=None, log_frequency=None, name=None):
        """
        Wraps the Python wrapper of BORG to make it compatible w/ Platypus.
        :param problem: Problem instance
        :param epsilons: objective space resolution
        :param log_frequency: frequency of output
        :param name: prefix string used for output filenames
        """
        super(BorgC, self).__init__(problem, evaluator=None, log_frequency=log_frequency)
        self.settings = {}
        if log_frequency is not None: self.settings['frequency'] = log_frequency
        if name is not None:
            self.settings['runtimefile'] = name+'_runtime.csv'
            self.name = f'{self.__class__.__name__}_{name}'
        self.name = name
        self.seed = seed
        self.nfe = 0
        self.result = None
        self.borg = None

        def problem_function(*vars):
            solution = Solution(problem)
            solution.variables[:] = vars
            solution.evaluate()
            constrs = [f(x) for (f, x) in zip(solution.problem.constraints, solution.constraints)]
            try:
                problem.function._progress_bar.update()
            except Exception as e:
                LOGGER.error(e, exc_info=True)
            return solution.objectives._data, constrs

        problem.borg_function = problem_function
        self.problem = problem
        platypus2borg_directions = {
            Problem.MAXIMIZE: Direction.MAXIMIZE,
            Problem.MINIMIZE: Direction.MINIMIZE
        }
        self.directions = [platypus2borg_directions[d] for d in problem.directions]

        '''
        solution_fields = ['variables', 'objectives']
        def problem_function_namedtuple(*vars):
            solution = namedtuple('Solution', solution_fields)
            solution.variables = vars
            solution.objectives = [0]*problem.nobjs
            problem.evaluate(solution)
            return solution.objectives
        '''

        if not isinstance(epsilons, list):
            epsilons = [epsilons]*problem.nobjs
        self.epsilons = epsilons

    def borg_init(self):
        import borg
        self.borg = borg
        if self.seed is not None:
            self.borg.Configuration.seed(self.seed)
        borg_obj = self.borg.Borg(self.problem.nvars, self.problem.nobjs, self.problem.nconstrs, self.problem.borg_function, directions=self.directions)
        borg_obj.setBounds(*[[vtype.min_value, vtype.max_value] for vtype in self.problem.types])
        borg_obj.setEpsilons(*self.epsilons)
        return borg_obj

    def borg_deinit(self):
        self.borg = None

    def borg_solve(self, borg_obj):
        raise NotImplementedError('Please use one of parent classes, either SerialBorgC or MpiBorgC')

    def borg_result(self, borg_res):
        result = None
        if borg_res:
            result = []
            for borg_sol in borg_res:
                s = Solution(self.problem)
                s.variables[:] = borg_sol.getVariables()
                s.evaluate()
                try:
                    np.testing.assert_array_almost_equal(s.objectives[:], borg_sol.getObjectives())
                except Exception as e:
                    LOGGER.warning(f'x = {s.variables[:]}, {e}')
                result.append(s)
        return result

    def step(self):
        # Initialize Borg object
        borg_obj = self.borg_init()
        # Solve
        borg_res = self.borg_solve(borg_obj)
        # Process results (if any)
        self.result = self.borg_result(borg_res)
        # Run deinit routines (e.g. StopMPI) & unload module
        self.borg_deinit()
        # Let Platypus know we're done
        self.nfe = self.settings['maxEvaluations']

    def run(self, condition):
        assert isinstance(condition, int)
        self.settings['maxEvaluations'] = condition
        super(BorgC, self).run(condition)


class SerialBorgC(BorgC):
    def borg_solve(self, borg_obj):
        # Init progress bar
        self.problem.function._progress_bar = tqdm(range(self.settings['maxEvaluations']))
        # Run BORG
        return borg_obj.solve(self.settings)


class MpiBorgC(BorgC):
    def __init__(self, problem, epsilons, seed=None, log_frequency=None, name=None):
        super(MpiBorgC, self).__init__(problem, epsilons=epsilons, seed=seed,
                                       log_frequency=log_frequency, name=name)
        # rename runtimefile -> runtime
        if name is not None:
            self.settings['runtime'] = self.settings['runtimefile']
            self.settings.pop('runtimefile')

    def borg_init(self):
        ret = super(MpiBorgC, self).borg_init()
        self.borg.Configuration.startMPI()
        return ret

    def borg_deinit(self):
        self.borg.Configuration.stopMPI()
        super(MpiBorgC, self).borg_deinit()

    def borg_solve(self, borg_obj):
        # Init progress bar
        self.problem.function._progress_bar = tqdm(range(self.settings['maxEvaluations']))
        # Run BORG
        borg_res = borg_obj.solveMPI(**self.settings)
        return borg_res


class ExternalBorgC(BorgC):
    def __init__(self, *args, mpirun=None, **kwargs):
        if mpirun == '-np 1':
            mpirun = None
        self.mpirun = mpirun
        self.loglevel = LOGGER.getEffectiveLevel()
        super(ExternalBorgC, self).__init__(*args, **kwargs)
        kwargs.pop('name', None)
        self.borg_kwargs = kwargs
        self.borg_args = args


    def borg_init(self):
        # Serialize algorithm to run
        import dill
        self.pickle = dill
        if self.mpirun is None:
            borg_class = SerialBorgC
            self.borg_launcher = []
        else:
            borg_class = MpiBorgC
            self.borg_launcher = ['mpirun',] + self.mpirun.split(' ')
        import os
        if self.name is None:
            passed_name = None
        else:
            passed_name = os.path.join(os.getcwd(), self.name)
        algo2dump = borg_class(*self.borg_args,
                               name=passed_name, **self.borg_kwargs)
        from tempfile import TemporaryDirectory
        self.tempdir = TemporaryDirectory()
        self.borg_algo_dumpfile = os.path.join(self.tempdir.name, 'algo.dmp')
        self.borg_runscript = os.path.join(self.tempdir.name, 'runscript.py')
        self.borg_result_dumpfile = os.path.join(self.tempdir.name, 'result.dmp')
        with open(self.borg_algo_dumpfile, 'wb') as f:
            self.pickle.dump(algo2dump, f)
        #LOGGER.info(f'Dumped {algo2dump.__class__.__name__} object to "{self.borg_algo_dumpfile}"')
        with open(self.borg_runscript, 'w') as f:
            f.write(f"""
import dill
import sys
import logging
import os
logging.basicConfig(level={self.loglevel})
sys.path.append("{os.getcwd()}")
os.environ['LD_LIBRARY_PATH'] = "{os.environ['LD_LIBRARY_PATH']}"
os.environ['DYLD_LIBRARY_PATH'] = "{os.environ['DYLD_LIBRARY_PATH']}"
#from pprint import pprint
#pprint(os.environ)
with open("{self.borg_algo_dumpfile}", "rb") as f:
    algo = dill.load(f)
try:
    algo.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
except:
    pass
algo.run({self.settings['maxEvaluations']})
if algo.result:
    with open("{self.borg_result_dumpfile}", "wb") as f:
        dill.dump(algo.result, f)                       
""")
        return self.borg_runscript

    def borg_solve(self, borg_obj):
        import subprocess
        cmd_args = self.borg_launcher + [sys.executable, self.borg_runscript,]
        cmd_args_string = ' '.join(cmd_args)
        LOGGER.info(f'Launching "{cmd_args_string}"')
        proc = subprocess.Popen(cmd_args)
        stdout, stderr = proc.communicate()
        with open(self.borg_result_dumpfile, 'rb') as f:
            res = self.pickle.load(f)
        logdisplay_endsection = '-'*len('INFO:Platypus:Borg output-----')
        if stdout is not None:
            LOGGER.info(f'Borg output-----\n{stdout}\n{logdisplay_endsection}')
        if stderr is not None:
            LOGGER.info(f'\n-----Borg error------\n{stderr}\n{logdisplay_endsection}')
        return res

    def borg_result(self, borg_res):
        return borg_res

    def borg_deinit(self):
        self.tempdir.cleanup()
        self.tempdir = None
        pass
