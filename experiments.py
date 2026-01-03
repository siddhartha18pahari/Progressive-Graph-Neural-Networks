from itertools import product
from rich.progress import Progress

def create_train_commands(registry: WandBJobRegistry):
    # --- Constants / lookups ---
    datasets = ['facebook', 'reddit', 'amazon', 'facebook-100', 'wenet']

    batch_size = {
        'facebook': 256,
        'reddit': 2048,
        'amazon': 4096,
        'facebook-100': 4096,
        'wenet': 1024,
    }
    max_degree = {
        'facebook': 100,
        'reddit': 400,
        'amazon': 50,
        'facebook-100': 100,
        'wenet': 400,
    }

    methods = ['progap', 'gap']
    levels = ['none', 'edge', 'node']

    # Shared defaults for all (dataset, method, level)
    base_defaults = {
        'hidden_dim': 16,
        'activation': 'selu',
        'optimizer': 'adam',
        'learning_rate': [0.01, 0.05],
        'repeats': 10,
        'epochs': 100,
        'batch_size': 'full',
        'verbose': False,
    }

    # Method-specific defaults
    progap_defaults = {
        'base_layers': [1, 2],
        'head_layers': 1,
        'jk': 'cat',
        'depth': [1, 2, 3, 4, 5],
        'layerwise': False,
    }

    gap_defaults = {
        'encoder_layers': 2,
        'base_layers': 1,
        'head_layers': 1,
        'combine': 'cat',
        'hops': [1, 2, 3, 4, 5],
    }

    # Node-level overrides applied to any method when level == "node"
    node_overrides = {
        'max_grad_norm': 1.0,
        'epochs': [5, 10],  # overrides base default epochs
    }

    # Privacy/accuracy epsilon grids
    eps_grid = {
        'node': [2, 4, 8, 16, 32],
        'edge': [0.25, 0.5, 1, 2, 4],
        'none': None,
    }

    def build_params(dataset: str, method: str, level: str) -> dict:
        """Construct params for one (dataset, method, level) without mutating shared dicts."""
        params = dict(base_defaults)

        # Method-specific
        if method == 'progap':
            params.update(progap_defaults)
        elif method == 'gap':
            params.update(gap_defaults)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Level-specific
        if level == 'node':
            params.update(node_overrides)
            params['max_degree'] = max_degree[dataset]
            params['batch_size'] = batch_size[dataset]
        return params

    def register_and_count(script: str, method: str, level: str, dataset: str, params: dict) -> int:
        """Register and return the number of newly-added jobs (best-effort)."""
        before = len(registry.df_job_cmds)
        registry.register(script, method, level, dataset=dataset, **params)
        after = len(registry.df_job_cmds)
        return after - before

    progress = Progress(
        *Progress.get_default_columns(),
        "[cyan]{task.fields[registered]}[/cyan] jobs registered",
        console=console,
    )
    task = progress.add_task("generating jobs", total=None, registered=0)

    registered = len(registry.df_job_cmds)

    with progress:
        for dataset in datasets:
            # ---------------------------------------------------------
            # 1) Accuracy/Privacy trade-off sweep (method x level)
            # ---------------------------------------------------------
            for method, level in product(methods, levels):
                params = build_params(dataset, method, level)

                # Only node/edge use epsilon sweeps here (same as original)
                if level in ('node', 'edge'):
                    params['epsilon'] = eps_grid[level]

                added = register_and_count('train.py', method, level, dataset, params)
                registered += added
                progress.update(task, registered=registered)

            # ---------------------------------------------------------
            # 2) Convergence (ProGAP, level != none)
            # ---------------------------------------------------------
            for level in levels:
                if level == 'none':
                    continue

                params = build_params(dataset, 'progap', level)
                params['repeats'] = 1
                params['depth'] = 5

                if level == 'node':
                    params['epsilon'] = 8
                    params['epochs'] = 10
                elif level == 'edge':
                    params['epsilon'] = 1
                    params['epochs'] = 100

                params['log_all'] = True

                added = register_and_count('train.py', 'progap', level, dataset, params)
                registered += added
                progress.update(task, registered=registered)

            # ---------------------------------------------------------
            # 3) Progressive vs. Layer-wise (ProGAP; register both)
            # ---------------------------------------------------------
            for level in levels:
                params_base = build_params(dataset, 'progap', level)

                if level == 'node':
                    params_base['epsilon'] = 8
                elif level == 'edge':
                    params_base['epsilon'] = 1

                # layerwise=True
                params_true = dict(params_base)
                params_true['layerwise'] = True
                added = register_and_count('train.py', 'progap', level, dataset, params_true)
                registered += added

                # layerwise=False
                params_false = dict(params_base)
                params_false['layerwise'] = False
                added = register_and_count('train.py', 'progap', level, dataset, params_false)
                registered += added

                progress.update(task, registered=registered)
