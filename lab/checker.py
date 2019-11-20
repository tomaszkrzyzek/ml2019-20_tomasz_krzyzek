from notebook import notebookapp
from typing import Callable
import urllib
import json
import ipykernel
import re
import os
import numpy as np
import torch

import utils
from utils import get_data


def checker(function: Callable = None, run_tests: bool = True, path: str = '.checker'):

    if function:
        return _Checker(function)
    else:
        def wrapper(function):
            return _Checker(function, run_tests=run_tests, path=path)

        return wrapper


class _Checker(object):

    def __init__(self, function: Callable = None, *,
                 run_tests: bool = True,
                 path: str = '.checker'):

        self.function = function
        self.base_dir = path
        self.run_tests = run_tests

        self.current_notebook_id = self.get_notebook_id()

        if not os.path.exists(os.path.join(self.base_dir, self.current_notebook_id)):
            print("Warning, not saved checker files, "
                  "make sure you pull the `.checker` folder from the repository")

        self.load_path = os.path.join(self.base_dir, self.current_notebook_id,
                                      self.function.__name__)

    def _load_inputs(self):
        if os.path.exists(self.load_path + '.in.npz'):
            inputs = dict(np.load(f"{self.load_path}.in.npz"))
            seed = inputs.pop('seed')
            return seed, inputs
        if os.path.exists(self.load_path + '.in.torch'):
            inputs = torch.load(f"{self.load_path}.in.torch")
            seed = inputs.pop('seed')
            return seed, inputs
        else:
            print("Warning! No saved checker files, "
                  "make sure you pull the `.checker` folder from the repository")

    def _load_outputs(self):
        if os.path.exists(self.load_path + '.out.npz'):
            outputs_dict = np.load(f"{self.load_path}.out.npz")
            if len(outputs_dict) == 1:
                return outputs_dict['0']
            elif len(outputs_dict) > 1:
                return tuple(outputs_dict.values())
        elif os.path.exists(self.load_path + '.out.torch'):
            outputs_dict = torch.load(f"{self.load_path}.out.torch")
            if len(outputs_dict) == 1:
                return outputs_dict['0']
            elif len(outputs_dict) > 1:
                return tuple(outputs_dict.values())
        else:
            print("Warning! No saved checker files, "
                  "make sure you pull the `.checker` folder from the repository")

    @staticmethod
    def _check_single(returned, expected):

        if isinstance(returned, list) and isinstance(expected, np.ndarray):
            returned = np.array(returned)

        if isinstance(expected, int):
            assert returned == expected, "Wrong value retuned!"
        elif isinstance(expected, np.ndarray):
            assert isinstance(returned, np.ndarray), f"Wrong type retuned: " \
                                                     f"{type(returned)}, expected: np.ndarray"
            assert returned.shape == expected.shape, "Wrong shape returned!"
            assert np.allclose(expected, returned), "Wrong value returned!"
        elif isinstance(expected, list):
            expected = np.array(expected)
            assert returned.shape == expected.shape, "Wrong shape returned!"
            assert np.allclose(expected, returned), "Wrong value returned!"
        elif isinstance(expected, torch.Tensor):
            assert isinstance(returned, torch.Tensor), f"Wrong type retuned: " \
                                                     f"{type(returned)}, expected: torch.Tensor"
            assert returned.shape == expected.shape, "Wrong shape returned!"
            assert torch.allclose(expected, returned), "Wrong value returned!"
        else:
            assert returned == expected, "Wrong value retuned!"

    def _check_results(self, returned, loaded):

        if isinstance(returned, tuple) and isinstance(loaded, tuple):
            assert len(returned) == len(loaded), f"Your function returned wrong " \
                                                 f"number of outputs: {len(returned)}, " \
                                                 f"expected: {len(loaded)}"

            for result, expected in zip(returned, loaded):
                self._check_single(result, expected)
        else:
            self._check_single(returned, loaded)

    def __call__(self, *args, **kwargs):

        if self.run_tests:
            seed, inputs = self._load_inputs()
            outputs = self._load_outputs()

            np.random.seed(seed)
            torch.manual_seed(seed)

            test_results = self.function(**inputs)
            self._check_results(test_results, outputs)

        # real run
        result = self.function(*args, **kwargs)

        return result

    @staticmethod
    def get_notebook_id():
        """Returns the absolute path of the Notebook or None if it cannot be determined
        NOTE: works only when the security is token-based or there is also no password
        """
        connection_file = os.path.basename(ipykernel.get_connection_file())
        kernel_id = connection_file.split('-', 1)[1].split('.')[0]

        for srv in notebookapp.list_running_servers():
            try:
                if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions')
                else:
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])
                sessions = json.load(req)
                for sess in sessions:
                    if sess['kernel']['id'] == kernel_id:
                        file_name = os.path.basename(
                            os.path.join(srv['notebook_dir'], sess['notebook']['path']))

                        m = re.match(r'\d\d', file_name)
                        if m:
                            return m.group()
                        else:
                            print(f"Couldn't parse notebook name: {file_name}")
            except:
                pass  # There may be stale entries in the runtime directory
        return None


def check_1_1(mean_error, mean_squared_error, max_error, train_sets):
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert np.isclose(mean_error(train_set_1d, np.array([8])), 8.897352)
    assert np.isclose(mean_error(train_set_2d, np.array([2.5, 5.2])), 7.89366)
    assert np.isclose(mean_error(train_set_10d, np.array(np.arange(10))), 14.16922)

    assert np.isclose(mean_squared_error(train_set_1d, np.array([3])), 23.03568)
    assert np.isclose(mean_squared_error(train_set_2d, np.array([2.4, 8.9])), 124.9397)
    assert np.isclose(mean_squared_error(train_set_10d, -np.arange(10)), 519.1699)

    assert np.isclose(max_error(train_set_1d, np.array([3])), 7.89418)
    assert np.isclose(max_error(train_set_2d, np.array([2.4, 8.9])), 14.8628)
    assert np.isclose(max_error(train_set_10d, -np.linspace(0, 5, num=10)), 23.1727)


def check_1_2(minimize_me, minimize_mse, minimize_max, train_set_1d):
    assert np.isclose(minimize_mse(train_set_1d), -0.89735)
    assert np.isclose(minimize_mse(train_set_1d * 2), -1.79470584)
    assert np.isclose(minimize_me(train_set_1d), -1.62603)
    assert np.isclose(minimize_me(train_set_1d ** 2), 3.965143)
    assert np.isclose(minimize_max(train_set_1d), 0.0152038)
    assert np.isclose(minimize_max(train_set_1d / 2), 0.007601903895526174)


def check_1_3(me_grad, mse_grad, max_grad, train_sets):
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert all(np.isclose(
        me_grad(train_set_1d, np.array([0.99])),
        [0.46666667]
    ))
    assert all(np.isclose(
        me_grad(train_set_2d, np.array([0.99, 8.44])),
        [0.21458924, 0.89772834]
    ))
    assert all(np.isclose(
        me_grad(train_set_10d, np.linspace(0, 10, num=10)),
        [-0.14131273, -0.031631, 0.04742431, 0.0353542, 0.16364242, 0.23353252,
         0.30958123, 0.35552034, 0.4747464, 0.55116738]
    ))

    assert all(np.isclose(
        mse_grad(train_set_1d, np.array([1.24])),
        [4.27470585]
    ))
    assert all(np.isclose(
        mse_grad(train_set_2d, np.array([-8.44, 10.24])),
        [-14.25378235,  21.80373175]
    ))
    assert all(np.isclose(
        max_grad(train_set_1d, np.array([5.25])),
        [1.]
    ))
    assert all(np.isclose(
        max_grad(train_set_2d, np.array([-6.28, -4.45])),
        [-0.77818704, -0.62803259]
    ))


def check_closest(fn):
    inputs = [
        (6, np.array([5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-2, np.array([-5, 12, 6, 0, -14, 3]))
    ]
    assert np.isclose(fn(*inputs[0]), 5), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[1]), 9), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[2]), 0), "Jest błąd w funkcji closest!"


def check_poly(fn):
    inputs = [
        (6, np.array([5.5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-5, np.array([6, 3, -12, 9, -15]))
    ]
    assert np.isclose(fn(*inputs[0]), 167.5), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[1]), 1539832), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[2]), -10809), "Jest błąd w funkcji poly!"


def check_multiplication_table(fn):
    inputs = [3, 5]
    assert np.all(fn(inputs[0]) == np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])), "Jest błąd w funkcji multiplication_table!"
    assert np.all(fn(inputs[1]) == np.array([
        [1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]
    ])), "Jest błąd w funkcji multiplication_table!"


def check_neg_log_likelihood(fn):
    inputs =     [
        (np.array([[5], [6.5], [14], [18], [-2], [3], [-5], [-8]]), np.array([5, 8])),
        (np.array([[-2.5], [0], [-5.5], [4.2], [-2.6], [3.9], [4.2], [9.9]]), np.array([7, -2]))
    ]
    assert np.isclose(fn(*inputs[0]), 28.473368724076067), "Jest błąd w funkcji log_likelihood!"
    assert np.isclose(fn(*inputs[1]), 65.56668571011694), "Jest błąd w funkcji log_likelihood!"


def check_grad_neg_log_likelihood(fn):
    inputs =     [
        (np.array([[5], [6.5], [14], [18], [-2], [3], [-5], [-8]]), np.array([5, 8])),
        (np.array([[-2.5], [0], [-5.5], [4.2], [-2.6], [3.9], [4.2], [9.9]]), np.array([7, -2]))
    ]
    desired_array = np.array([0.1328125, -0.12158203])
    assert all(np.isclose(np.ravel(fn(*inputs[0])), desired_array)), "Jest błąd w funkcji grad_log_likelihood!"
    desired_array = np.array([11.1, 48.67])
    assert all(np.isclose(np.ravel(fn(*inputs[1])), desired_array)), "Jest błąd w funkcji grad_log_likelihood!"

def check_02_pca(pca_cls):
    from sklearn import datasets
    os.makedirs(".checker/02/", exist_ok=True)

    np.random.seed(55)
    input_data = np.random.normal(5, 22, size=(1000, 4))
    np.random.seed(33)
    pca = pca_cls(n_components=3)
    pca.fit(input_data)
    returned = pca.transform(input_data)
    # np.savez_compressed(file=".checker/05/pca_random.out", data=returned)
    expected = np.load(".checker/05/pca_random.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong value returned!"

    input_data = datasets.load_iris().data
    np.random.seed(23)
    pca = pca_cls(var_to_explain=0.97)
    pca.fit(input_data)
    returned = pca.transform(input_data)
    # np.savez_compressed(file=".checker/05/pca_iris.out", data=returned)
    expected = np.load(".checker/05/pca_iris.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong value returned!"

    input_data = datasets.load_digits().data
    np.random.seed(27)
    pca = pca_cls(var_to_explain=0.90)
    pca.fit(input_data)
    returned = pca.transform(input_data)
    # np.savez_compressed(file=".checker/05/pca_digits.out", data=returned)
    expected = np.load(".checker/05/pca_digits.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong value returned!"

def check_02_linear_regression(lr_cls):
    from sklearn import datasets
    os.makedirs(".checker/02/", exist_ok=True)

    input_dataset = datasets.load_boston()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/lr_boston.out.npz", data=returned)
    expected = np.load(".checker/05/lr_boston.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 24.166099, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

    input_dataset = datasets.load_diabetes()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/lr_diabetes.out.npz", data=returned)
    expected = np.load(".checker/05/lr_iris.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 26004.287402, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

def check_02_regularized_linear_regression(lr_cls):
    from sklearn import datasets
    os.makedirs(".checker/02/", exist_ok=True)

    np.random.seed(54)
    input_dataset = datasets.load_boston()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/rlr_boston.out.npz", data=returned)
    expected = np.load(".checker/05/rlr_boston.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 42.8331406942, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

    np.random.seed(58)
    input_dataset = datasets.load_diabetes()
    lr = lr_cls(lr=1e-2, alpha=1e-4)
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/rlr_diabetes.out.npz", data=returned)
    expected = np.load(".checker/05/rlr_diabetes.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 26111.08336411, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

def check_07_logistic_reg(lr_cls):
    np.random.seed(10)
    torch.manual_seed(10)
    
    # **** First dataset ****
    input_dataset = utils.get_classification_dataset_1d()
    lr = lr_cls(1)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-3, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/07/lr_dataset_1d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 0.5098415017127991, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"
    
    preds_proba = lr.predict_proba(input_dataset.data)
    save_path = ".checker/07/lr_dataset_1d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"
    
    preds = lr.predict(input_dataset.data)
    save_path = ".checker/07/lr_dataset_1d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    # **** Second dataset ****
    input_dataset = utils.get_classification_dataset_2d()
    lr = lr_cls(2)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-2, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/07/lr_dataset_2d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 0.044230662286281586, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"
    
    preds_proba = lr.predict_proba(input_dataset.data)
    save_path = ".checker/07/lr_dataset_2d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"
    
    preds = lr.predict(input_dataset.data)
    save_path = ".checker/07/lr_dataset_2d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

def test_cv(cv_func):
    
    X, y = get_data(split=False)
    k = 5
    
    splits = cv_func(X=X, y=y, k=k)
    
    assert len(splits) == k, "Wrong number of splits returned!"
    
    for split in splits:
        assert len(split) == 4, f"Wrong tuple length for a single split, expected 4, got: {len(split)}"
        assert isinstance(split[0], np.ndarray), "Wrong type returned!"
        assert isinstance(split[1], np.ndarray), "Wrong type returned!"
        assert isinstance(split[2], np.ndarray), "Wrong type returned!"
        assert isinstance(split[3], np.ndarray), "Wrong type returned!"
        
        assert len(split[0]) == len(split[1]), f"Train split shape mismatch: {len(split[0])} and {len(split[1])}"
        assert len(split[2]) == len(split[3]), f"Test split shape mismatch: {len(split[0])} and {len(split[1])}"
        
    assert sum([split[2].shape[0] for split in splits]) == X.shape[0], \
        "Sum of all split lengths is mismatched with whole dataset, did you use whole dataset for splitting?"
                                                                                                     
                                                                            
from types import SimpleNamespace
from torch.optim import SGD
from torch.optim import Adagrad as torch_adagrad
from torch.optim import RMSprop as torch_rmsprop
from torch.optim import Adadelta as torch_adadelta
from torch.optim import Adam as torch_adam

def optim_f(w):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w ** 2)

def optim_g(w, b):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w + b)

opt_checker_1 = SimpleNamespace(f=optim_f, 
                                params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)])
opt_checker_2 = SimpleNamespace(f=optim_g, 
                               params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True),
                                       torch.tensor([1, -1], dtype=torch.float, requires_grad=True)])


test_params = {'Momentum': {'torch_cls': SGD,
                            'torch_params': {'lr': 0.1, 'momentum': 0.9},
                            'params': {'learning_rate': 0.1, 'gamma': 0.9}},
              'Adagrad': {'torch_cls': torch_adagrad,
                            'torch_params': {'lr': 0.5, 'eps': 1e-8},
                            'params': {'learning_rate': 0.5, 'epsilon': 1e-8}},
              'RMSProp': {'torch_cls': torch_rmsprop,
                          'torch_params': {'lr': 0.5, 'alpha': 0.9, 'eps': 1e-08,},
                          'params': {'learning_rate': 0.5, 'gamma': 0.9, 'epsilon': 1e-8}},
              'Adadelta': {'torch_cls': torch_adadelta,
                          'torch_params': {'rho': 0.9, 'eps': 1e-1},
                          'params': {'gamma': 0.9, 'epsilon': 1e-1}},
              'Adam': {'torch_cls': torch_adam,
                          'torch_params': {'lr': 0.5, 'betas': (0.9, 0.999), 'eps': 1e-08},
                          'params': {'learning_rate': 0.5, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}}}

def test_optimizer(optim_cls):
               
    test_dict = test_params[ optim_cls.__name__]
    
    for ns in [opt_checker_1, opt_checker_2]:
        
        torch_params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        
        torch_opt = test_dict['torch_cls'](torch_params, **test_dict['torch_params'])
        torch_opt.zero_grad()
        
        loss = ns.f(*torch_params)
        loss.backward()
        torch_opt.step()
        
        params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        
        opt = optim_cls(params, **test_dict['params'])
        opt.zero_grad()
    
        loss = ns.f(*params)
        loss.backward()
        opt.step()
        
        for p, tp in zip(params, torch_params):
            assert torch.allclose(p, tp)
