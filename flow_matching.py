import torch

class FlowMatching:
    def __init__(self, mode='ot'):
        self.mode = mode

    def sample_train_target(self, x1):
        B, E = x1.shape
        t = torch.rand(B, E, device=x1.device)
        x0 = torch.randn_like(x1)

        if self.mode == 'ot':
            # Optimal Transport Flow Matching (straight paths)
            xt = (1.0 - t) * x0 + t * x1
            vt_target = x1 - x0

        elif self.mode == 'vanilla':
            # Vanilla Flow Matching (independent coupling)
            xt = (1.0 - t) * x0 + t * x1
            vt_target = x1 - x0
            # Difference is in batch coupling conceptually, mathematically similar here for single pairs

        elif self.mode == 'diffusion':
            # Simplified DDPM-like formulation
            alpha_t = 1.0 - t
            xt = torch.sqrt(alpha_t) * x1 + torch.sqrt(1 - alpha_t) * x0
            vt_target = x0 # Predicting noise

        return xt, t, vt_target

    @torch.no_grad()
    def sample_inference(self, model, edges_feat, steps=50):
        E = edges_feat.shape[0]
        device = edges_feat.device

        xt = torch.randn(1, E, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.ones(1, E, device=device) * (i / steps)
            vt_pred = model(edges_feat, xt, t)

            if self.mode in ['ot', 'vanilla']:
                xt = xt + vt_pred * dt
            elif self.mode == 'diffusion':
                # Simplified Euler step for DDPM
                alpha_t = 1.0 - (i/steps)
                xt = xt - vt_pred * (dt / torch.sqrt(1 - alpha_t + 1e-5))

        return xt.squeeze(0)
