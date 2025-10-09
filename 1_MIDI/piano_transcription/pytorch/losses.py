import torch
import torch.nn.functional as F


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss

def moderate_onset_prioritization_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = 1.6 * onset_loss + 0.9 * offset_loss + 1.25 * frame_loss + 1.15 * velocity_loss
    return total_loss

def simple_onset_prioritization_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = 1.2 * onset_loss + 1.0 * offset_loss + 1.0 * frame_loss + 1.0 * velocity_loss
    return total_loss

def regress_onset_offset_frame_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    total_loss = onset_loss + offset_loss + frame_loss
    return total_loss

def regress_onset_offset_frame_velocity_laplace(model, output_dict, target_dict):
    """High-resolution note loss with Laplace NLL for onset/offset timing."""
    FIXED_B = 0.035  # 50 ms in normalized hop-units (since hop size = 1s)

    def laplace_nll(pred, target, mask=None, b=FIXED_B):
        if not torch.is_tensor(b):
            b = torch.tensor(b, device=pred.device, dtype=pred.dtype)
        nll = torch.log(2 * b) + (target - pred).abs() / b
        if mask is not None:
            nll = nll * mask
            return nll.sum() / mask.sum().clamp_min(1e-8)
        return nll.mean()

    mask_all   = target_dict['mask_roll']
    onset_mask = target_dict['onset_roll']
    offset_mask= target_dict['offset_roll']

    onset_loss = laplace_nll(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], onset_mask)
    offset_loss = laplace_nll(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], offset_mask)
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], mask_all)

    return onset_loss + offset_loss + frame_loss


def regress_onset_offset_frame_50ms_accuracy(model, output_dict, target_dict):
    """Custom loss function that directly optimizes for ±50ms onset/offset accuracy.
    
    This loss function is designed to maximize the percentage of notes that have 
    onset and offset predictions within ±50ms of the ground truth.
    """
    import torch.nn.functional as F
    
    # 50ms tolerance in normalized units (assuming 100 fps, so 50ms = 5 frames)
    TOLERANCE_MS = 50.0
    FRAMES_PER_SECOND = 100.0  # From config
    TOLERANCE_FRAMES = TOLERANCE_MS / 1000.0 * FRAMES_PER_SECOND  # 5.0 frames
    
    def accuracy_loss(pred, target, mask=None):
        """Loss that encourages predictions to be within ±50ms of targets."""
        # Calculate absolute error
        abs_error = torch.abs(pred - target)
        
        # Create binary mask for predictions within tolerance
        within_tolerance = (abs_error <= TOLERANCE_FRAMES).float()
        
        # Apply mask if provided
        if mask is not None:
            within_tolerance = within_tolerance * mask
            valid_predictions = mask.sum().clamp_min(1e-8)
        else:
            valid_predictions = within_tolerance.numel()
        
        # Calculate percentage within tolerance (higher is better, so we minimize 1 - percentage)
        accuracy_percentage = within_tolerance.sum() / valid_predictions
        
        # We want to maximize accuracy, so minimize (1 - accuracy)
        accuracy_loss = 1.0 - accuracy_percentage
        
        # Add a small MSE component to encourage precise predictions within the tolerance
        mse_component = F.mse_loss(pred, target, reduction='none')
        if mask is not None:
            mse_component = mse_component * mask
            mse_loss = mse_component.sum() / mask.sum().clamp_min(1e-8)
        else:
            mse_loss = mse_component.mean()
        
        # Combine accuracy loss with MSE (weighted)
        total_loss = accuracy_loss + 0.1 * mse_loss
        
        return total_loss
    
    # Get masks for onset and offset predictions
    onset_mask = target_dict['onset_roll']
    offset_mask = target_dict['offset_roll']
    frame_mask = target_dict['mask_roll']
    
    # Calculate losses
    onset_loss = accuracy_loss(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], onset_mask)
    offset_loss = accuracy_loss(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], offset_mask)
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], frame_mask)
    
    # Combine losses (you can adjust these weights)
    total_loss = onset_loss + offset_loss + frame_loss
    
    return total_loss



def regress_pedal_bce(model, output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce

    elif loss_type == "regress_onset_offset_frame_bce":
        return regress_onset_offset_frame_bce

    elif loss_type == "regress_onset_offset_frame_velocity_laplace":
        return regress_onset_offset_frame_velocity_laplace

    elif loss_type == "regress_onset_offset_frame_50ms_accuracy":
        return regress_onset_offset_frame_50ms_accuracy
    
    elif loss_type == "moderate_onset_prioritization_bce":
        return moderate_onset_prioritization_bce
    elif loss_type == "simple_onset_prioritization_bce":
        return simple_onset_prioritization_bce

    else:
        raise Exception('Incorrect loss_type!')