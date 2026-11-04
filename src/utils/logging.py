"""Logging utilities for WandB and TensorBoard."""

from typing import Optional, Dict, Any
import os

from src.utils.env_config import env_str


class Logger:
    """Unified logger interface for WandB and TensorBoard."""
    
    def __init__(
        self,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = env_str('WANDB_PROJECT', 'WandB project name',
                                    required=False, default='gpt-transformer'),
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.wandb_run = None
        self.tb_writer = None
        
        # Initialize WandB
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config
                )
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
        
        # Initialize TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join('runs', project_name)
                if run_name:
                    log_dir = os.path.join(log_dir, run_name)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: tensorboard not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False
    
    def log(self, metrics: Dict[str, float], step: int, epoch: Optional[int] = None):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step
            epoch: Epoch number (optional)
        """
        if self.use_wandb and self.wandb_run:
            log_dict = {**metrics}
            if epoch is not None:
                log_dict['epoch'] = epoch
            self.wandb_run.log(log_dict, step=step)
        
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
            if epoch is not None:
                self.tb_writer.add_scalar('epoch', epoch, step)
    
    def close(self):
        """Close loggers."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()


def setup_logging(
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    project_name: str = 'gpt-transformer',
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Logger:
    """Setup logging.
    
    Args:
        use_wandb: Whether to use WandB
        use_tensorboard: Whether to use TensorBoard
        project_name: Project name
        run_name: Run name
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    return Logger(
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        project_name=project_name,
        run_name=run_name,
        config=config
    )


def log_metrics(
    logger: Logger,
    metrics: Dict[str, float],
    step: int,
    epoch: Optional[int] = None
):
    """Log metrics using logger.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        step: Global step
        epoch: Epoch number (optional)
    """
    logger.log(metrics, step=step, epoch=epoch)

