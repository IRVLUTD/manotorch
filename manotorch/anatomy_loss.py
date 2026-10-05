import math

import torch
from torch import nn


class AnatomyConstraintLossEE(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._setup = False
        self._eps = 1e-6
        self.reduction = reduction
        # Output order is MCP fingers, PIP fingers, DIP fingers, then the three thumb joints.
        self.register_buffer("_joint_ids", torch.tensor([1, 4, 10, 7, 2, 5, 11, 8, 3, 6, 12, 9, 13, 14, 15]),
                             persistent=False)
        self.register_buffer("_lower", torch.empty(0, dtype=torch.float64), persistent=False)
        self.register_buffer("_upper", torch.empty(0, dtype=torch.float64), persistent=False)

    @staticmethod
    def _validate_config(config):
        config = list(config)
        if len(config) != 3:
            raise ValueError("Each joint limit needs three entries: twist, spread, bend")
        for entry in config:
            try:
                parts = entry.split(",")
                expected = ["+-"] if len(parts) == 1 else ["+", "-"]
                if len(parts) != len(expected):
                    raise ValueError
                for part, sign in zip(parts, expected, strict=True):
                    label, value = part.split(":")
                    tolerance = float(value)
                    if label != sign or not math.isfinite(tolerance) or tolerance < 0:
                        raise ValueError
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid angle limit {entry!r}; use '+-:45' or '+:45,-:15'") from error
        return config

    def setup(
        self,
        thumb_cmc=("+-:45", "+:45,-:15", "+:45,-:0"),
        thumb_mcp=("+-:0", "+-:10", "+:90,-:0"),
        thumb_pip=("+-:0", "+-:0", "+:90,-:0"),
        finger_mcp=("+-:0", "+-:5", "+:90,-:0"),
        finger_pip=("+-:0", "+-:0", "+:90,-:0"),
        finger_dip=("+-:0", "+-:0", "+:90,-:0"),
    ):
        """Setup the angle degree limit for each types of joints.

        Args:
            thumb_cmc (list, optional): The angle limies of thumb CMC joint. 
                Defaults to ["+-:45", "+:45,-:15", "+:45,-:0"], for angles in 
                twist, spread, and bend direction, respectively.
            thumb_mcp (list, optional):  Defaults to ["+-:0", "+-:10", "+:90,-:0"].
            thumb_pip (list, optional):  Defaults to ["+-:0", "+-:0", "+:90,-:0"].
            finger_mcp (list, optional): Defaults to ["+-:0", "+-:5", "+:90,-:0"].
            finger_pip (list, optional): Defaults to ["+-:0", "+-:0", "+:90,-:0"].
            finger_dip (list, optional): Defaults to ["+-:0", "+-:0", "+:90,-:0"].
        """
        configs = {name: self._validate_config(cfg) for name, cfg in {
            "thumb_cmc": thumb_cmc, "thumb_mcp": thumb_mcp, "thumb_pip": thumb_pip,
            "finger_mcp": finger_mcp, "finger_pip": finger_pip, "finger_dip": finger_dip,
        }.items()}
        for name, cfg in configs.items():
            setattr(self, name, cfg)

        ordered = [configs[name] for name in ["finger_mcp"] * 4 + ["finger_pip"] * 4 + ["finger_dip"] * 4
                   + ["thumb_cmc", "thumb_mcp", "thumb_pip"]]
        lower, upper = [], []
        for joint in ordered:
            lo, hi = [], []
            for axis in joint:
                if axis.startswith("+-:"):
                    positive = negative = float(axis[3:])
                else:
                    pos, neg = axis.split(",")
                    positive, negative = float(pos[2:]), float(neg[2:])
                # Match the legacy operation order, including its exact ReLU boundary rounding.
                lo.append(-(negative / 180.0) * math.pi)
                hi.append((positive / 180.0) * math.pi)
            lower.append(lo)
            upper.append(hi)
        # Keep double precision while compiling limits; converting to the input dtype happens once per device/dtype.
        self._lower = torch.tensor(lower, dtype=torch.float64, device=self._lower.device)
        self._upper = torch.tensor(upper, dtype=torch.float64, device=self._upper.device)
        self._config_key = self._current_config_key()

        self._setup = True

    def _current_config_key(self):
        return tuple(tuple(getattr(self, name)) for name in
                     ("thumb_cmc", "thumb_mcp", "thumb_pip", "finger_mcp", "finger_pip", "finger_dip"))

    def _apply(self, fn, **kwargs):
        result = super()._apply(fn, **kwargs)
        if self._setup:
            # Module dtype conversions can round cached limits; regenerate from the degree strings.
            self._config_key = None
        return result

    def _cal_loss_one_axis(self, ee, cfg):
        """Calculate the anatomy loss for one axis.

        Args:
            ee (torch.Tensor): (B, NJ), the euler angle of a certain axis. 
                NJ is number of joints that has this type of axis.
                NJ = 1 for thumb, NJ = 4 for other joints.
            cfg (str): The configuration of the angle limit on this axis.

        Returns:
            torch.Tensor: (B, NJ)
        """
        non_zero_mask = torch.abs(ee) > self._eps

        if "+-" in cfg:
            _, tolerance = cfg.split(":")
            tol = (float(tolerance) / 180.0) * torch.pi
            loss = torch.relu(torch.abs(ee) - tol)
        else:
            pos_cfg, neg_cfg = cfg.split(",")
            pos_tolerance, neg_tolerance = pos_cfg.split(":")[1], neg_cfg.split(":")[1]

            pos_tol = (float(pos_tolerance) / 180.0) * torch.pi
            neg_tol = (float(neg_tolerance) / 180.0) * torch.pi
            neg_mask = ee < 0
            pos_mask = ~neg_mask
            loss = torch.relu(-ee - neg_tol) * neg_mask.float() + \
                  torch.relu(ee - pos_tol) * pos_mask.float()

        loss = loss * non_zero_mask.float()
        return loss

    def _cal_loss_one_joint(self, ee, cfg):
        """Calculate the anatomy loss for one joint.
        as the sum of twist, spread, bend angles.

        Args:
            ee (torch.Tensor): (B, NJ, 3)
            cfg (str): config string for that joint type, e.g. ["+-:0", "+-:0", "+:90,-:0"]

        Returns:
            torch.Tensor: (B, NJ)
        """
        twist_loss = self._cal_loss_one_axis(ee[:, :, 0], cfg[0])
        spread_loss = self._cal_loss_one_axis(ee[:, :, 1], cfg[1])
        bend_loss = self._cal_loss_one_axis(ee[:, :, 2], cfg[2])

        spv = twist_loss + spread_loss + bend_loss
        return spv

    def forward(self, euler_angles, **kwargs):
        """Calculate the anatomy loss for the given euler angles.
        #  the euler-angles' order of the right hand:
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/ 

        Args:
            euler_angles (torch.Tensor): (B, NJ, 3), the euler angles of the joints.

        Raises:
            ValueError: If the setup function is not called before.

        Returns:
            torch.Tensor: a scalar tensor, the anatomy loss.
        """

        if self._setup is False:
            raise ValueError("Please setup the angle limit first.")

        if self._current_config_key() != self._config_key or self._lower.dtype != euler_angles.dtype:
            # Rebuild from strings after dtype changes, so a prior float32 call does not truncate float64 limits.
            # Preserve callers that edit public lists; configuration changes are initialization work.
            names = ("thumb_cmc", "thumb_mcp", "thumb_pip", "finger_mcp", "finger_pip", "finger_dip")
            self.setup(**{name: getattr(self, name) for name in names})
        # Lazy device placement keeps the historical loss(angles.cuda()) usage working. Steady-state calls do no copies.
        self._joint_ids = self._joint_ids.to(euler_angles.device)
        self._lower = self._lower.to(euler_angles)
        self._upper = self._upper.to(euler_angles)
        ee = euler_angles.index_select(1, self._joint_ids)
        per_axis = torch.relu(ee - self._upper) + torch.relu(self._lower - ee)
        per_axis = per_axis * (ee.abs() > self._eps).float()
        loss_all = per_axis.sum(-1)  # (B, 15)
        if self.reduction == "none":
            return loss_all
        elif self.reduction == "mean":
            return loss_all.mean()
        elif self.reduction == "sum":
            return loss_all.sum()
        else:
            raise ValueError("Unknown reduction type.")
