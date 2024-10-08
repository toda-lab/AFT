from aft import AFT
from FairnessTestMethods import Vbtx, Vbt, Themis
from utils.BlackBoxModel import BlackBoxModel
import multiprocessing
from joblib import load
import argparse


def exp(dataset_name, model_name, protected_pair, method, runtime, repeat, show_logging, res_logging=False, repeat_label=None, start_label=0):
    protected_name = protected_pair[0]
    protected_param = protected_pair[1]

    # Create CuT (Classifier under Test).
    # Note that here, a CuT (Component Under Test) is a black-box model.
    # This means we can only:
    # (1) Know the number of input attributes for the CuT and the range of values for each attribute.
    # (2) Given an input, use the CuT to predict its prediction outcome.
    # The range of values for each attribute is obtained by analyzing the training data of the CuT in ./Datasets.
    dataset_csv = "GermanCredit" if dataset_name == "Credit" else dataset_name
    data_range, df = BlackBoxModel.create_data_range_from_csv(f"./Datasets/{dataset_csv}.csv")
    MODEL_ = load(f"FairnessTestCases/{model_name}{dataset_name}.joblib")
    CuT = BlackBoxModel(data_range, MODEL_, feature_list=df.columns.tolist())

    # Perform IFT (individual fairness testing).
    # Each IFT algorithm requires two inputs:
    # (1) a CuT, and
    # (2) a protected attribute, including its name (protected_name) and its index (protected_param).
    for _ in range(start_label, start_label + repeat):
        tester = None
        if repeat_label is not None:
            _ = repeat_label

        if method == "aft":
            tester = AFT(CuT, [protected_param], no_train_data_sample=5000, show_logging=show_logging)
            tester.test(runtime=runtime, max_leaf_nodes=1000,
                        max_train_data_each_path=10, max_sample_each_path=100, MaxDiscPathPair=100, MaxTry=10000,
                        dt_search_mode="random+flip", check_type="themis",
                        label=(f"{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}", _))

        elif method == "vbtx":
            vbtx_ver = "improved"
            tester = Vbtx(CuT, [protected_param], no_train_data_sample=5000,
                          vbtx_ver=vbtx_ver, show_logging=show_logging)
            tester.test(runtime=runtime, label=(f"{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}", _))

        elif method == "vbt":
            vbtx_ver = "vbt"
            tester = Vbt(CuT, [protected_param], no_train_data_sample=5000,
                          vbtx_ver=vbtx_ver, show_logging=show_logging)
            tester.test(runtime=runtime, label=(f"{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}", _))

        elif method == "themis":
            tester = Themis(CuT, [protected_param], show_logging=show_logging)
            tester.test(runtime=runtime, max_test=None, max_disc=None,
                        label=(f"{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}", _))

        else:
            print(f"No method called {method}")

        if res_logging:
            option_reports = ""
            with open("log.txt", 'a') as file:
                file.write(
                    f"{method}-{model_name}-{dataset_name}-{protected_name}-{_},{tester.no_disc},{tester.no_test},{tester.no_disc / float(tester.no_test + .0001)},{tester.cpu_time_consumed},{tester.real_time_consumed}{option_reports}\n")


def para_exp_main(runtime, repeat, repeat_run_together=True):
    check_models = ["LogReg", "RanForest", "DecTree", "MLP"]
    dataset_names = ["Adult", "Credit", "Bank"]
    method_list = ["aft", "vbtx", "vbt", "themis"]
    start_label = 0
    paras = list()
    for method in method_list:
        for dataset_name in dataset_names:
            protected_list = list()
            if dataset_name == "Adult":
                protected_list = [("sex", 8), ("race", 7), ("age", 0)]
            elif dataset_name == "Bank":
                protected_list = [("age", 0)]
            elif dataset_name == "Credit":
                protected_list = [("sex", 8), ("age", 12)]
            for protected_pair in protected_list:
                for model_name in check_models:
                    if repeat_run_together:
                        paras.append((dataset_name, model_name, protected_pair, method, runtime, repeat, False, True, None, start_label))
                    else:
                        for _ in range(repeat):
                            paras.append((dataset_name, model_name, protected_pair, method, runtime, 1, False, True, _, 0))
    pool = multiprocessing.Pool(processes=12)
    pool.starmap(exp, paras)
    pool.close()
    pool.join()


def print_usage():
    print("dataset and protected_attr pairs: (Adult,sex), (Adult,race), (Adult,age), (Credit,sex), (Credit,age) (Bank,age)")


def get_attr_index(dataset_name, protected_attr):
    if dataset_name == "Adult":
        if protected_attr == "sex":
            attr_index = 8
        elif protected_attr == "race":
            attr_index = 7
        elif protected_attr == "age":
            attr_index = 0
        else:
            print(f"no protected attribute called {protected_attr}")
            print_usage()
            exit()
    elif dataset_name == "Bank":
        if protected_attr == "age":
            attr_index = 0
        else:
            print(f"no protected attribute called {protected_attr}")
            print_usage()
            exit()
    elif dataset_name == "Credit":
        if protected_attr == "sex":
            attr_index = 8
        elif protected_attr == "age":
            attr_index = 12
        else:
            print(f"no protected attribute called {protected_attr}")
            print_usage()
            exit()
    else:
        print(f"no dataset called {dataset_name}")
        print_usage()
        exit()
    return attr_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments based on specified parameters")
    parser.add_argument("all", nargs='?',  const=True, default=False,
                        help="Specify 'all' to run all experiments")
    parser.add_argument("--dataset_name", choices=["Adult", "Credit", "Bank"], default="Adult",
                        help="Name of the dataset (default: Adult)")
    parser.add_argument("--protected_attr", choices=["sex", "race", "age"], default="sex",
                        help="Name of the protected attribute (default: sex)")
    parser.add_argument("--runtime", type=int, default=3600,
                        help="Runtime in seconds (default: 3600)")
    parser.add_argument("--repeat", type=int, default=30,
                        help="Number of repetitions (default: 30)")
    parser.add_argument("--method", choices=["aft", "vbtx", "vbt", "themis"],
                        default="aft", help="Method to be used (default: aft)")
    parser.add_argument("--model_name", choices=["LogReg", "RanForest", "DecTree", "MLP"],
                        default="MLP", help="Model name (default: MLP)")
    parser.add_argument("--disable_log", action="store_false", dest="enable_log",
                        help="Disable logging (default: enable)")

    args = parser.parse_args()

    if args.all:
        para_exp_main(3600, 30, repeat_run_together=True)
    else:
        attr_index = get_attr_index(args.dataset_name, args.protected_attr)
        print(f"Running experiment with parameters:\n"
              f"Dataset Name: {args.dataset_name}\n"
              f"Protected Attribute: {args.protected_attr}\n"
              f"Runtime: {args.runtime} seconds\n"
              f"Repeat: {args.repeat} times\n"
              f"Method: {args.method}\n"
              f"Model Name: {args.model_name}")
        exp(dataset_name=args.dataset_name, model_name=args.model_name,
            protected_pair=(args.protected_attr, attr_index), method=args.method, runtime=args.runtime,
            repeat=args.repeat, show_logging=args.enable_log, res_logging=False, repeat_label=None)

