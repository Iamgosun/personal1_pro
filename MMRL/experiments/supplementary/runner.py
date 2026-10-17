"""Optional native trainer hooks: initialization snapshot, or E1 epoch snapshots.

No extra model forward or posterior draw; therefore no training RNG perturbation.
These extend RefactorRunner and leave all native validation/training unchanged.
"""


def register_trajectory():
    from dassl.engine import TRAINER_REGISTRY
    from trainers.refactor_runner import RefactorRunner
    names=(TRAINER_REGISTRY.registered_names() if hasattr(TRAINER_REGISTRY,'registered_names')
           else getattr(TRAINER_REGISTRY,'_obj_map',{}))
    if 'SupplementInitRunner' not in names:
        @TRAINER_REGISTRY.register()
        class SupplementInitRunner(RefactorRunner):
            def before_train(self):
                super().before_train()
                from pathlib import Path
                root=Path(self.cfg.OUTPUT_DIR)/'trajectory'
                # Lightweight checkpoint epoch=0; before any optimizer update.
                self.save_model(-1,str(root),is_best=False)
    else:
        SupplementInitRunner=TRAINER_REGISTRY.get('SupplementInitRunner')
    if 'SupplementTrajectoryRunner' not in names:
        @TRAINER_REGISTRY.register()
        class SupplementTrajectoryRunner(SupplementInitRunner):
            def after_epoch(self):
                super().after_epoch()
                from pathlib import Path
                root=Path(self.cfg.OUTPUT_DIR)/'trajectory'
                self.save_model(self.epoch,str(root),is_best=False)
