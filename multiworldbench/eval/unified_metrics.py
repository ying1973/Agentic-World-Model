import math


def unified_video_alignment_score(
    fvd: float,
    lpips: float,
    ssim: float,
    psnr: float,
    *,
    # ---- normalization hyperparameters ----
    fvd_tau: float = 2000.0,   # median/mean FVD on validation set
    ssim_gamma: float = 2.0,   # SSIM enhancement factor
    psnr_mu: float = 15.0,     # PSNR mid-point
    psnr_sigma: float = 2.0,   # PSNR smoothness
    # ---- metric weights ----
    w_fvd: float = 0.25,
    w_lpips: float = 0.25,
    w_ssim: float = 0.25,
    w_psnr: float = 0.25,
) -> float:
    """
    Compute unified perceptual alignment score for video-to-video evaluation.
    Output range: [0, 100], higher is better.
    """

    # ---------- 1. FVD (lower is better) ----------
    # Exponential decay to suppress long-tail effect
    s_fvd = math.exp(-fvd / fvd_tau)

    # ---------- 2. LPIPS (lower is better) ----------
    s_lpips = 1.0 - lpips
    s_lpips = max(0.0, min(1.0, s_lpips))  # safety clamp

    # ---------- 3. SSIM (higher is better) ----------
    s_ssim = max(0.0, min(1.0, ssim)) ** ssim_gamma

    # ---------- 4. PSNR (higher is better) ----------
    s_psnr = 1.0 / (1.0 + math.exp(-(psnr - psnr_mu) / psnr_sigma))

    # ---------- 5. Weighted aggregation ----------
    total_weight = w_fvd + w_lpips + w_ssim + w_psnr
    assert abs(total_weight - 1.0) < 1e-6, "Metric weights must sum to 1."

    perceptual_score = (
        w_fvd * s_fvd +
        w_lpips * s_lpips +
        w_ssim * s_ssim +
        w_psnr * s_psnr
    )

    return 100.0 * perceptual_score



def unified_image_alignment_score(
    fid: float,
    lpips: float,
    ssim: float,
    psnr: float,
    *,
    # ---- normalization hyperparameters ----
    fid_tau: float = 50.0,   # median/mean FVD on validation set
    ssim_gamma: float = 2.0,   # SSIM enhancement factor
    psnr_mu: float = 16.0,     # PSNR mid-point
    psnr_sigma: float = 2.0,   # PSNR smoothness
    # ---- metric weights ----
    w_fid: float = 0.25,
    w_lpips: float = 0.25,
    w_ssim: float = 0.25,
    w_psnr: float = 0.25,
) -> float:
    """
    Compute unified perceptual alignment score for video-to-video evaluation.
    Output range: [0, 100], higher is better.
    """

    # ---------- 1. FVD (lower is better) ----------
    # Exponential decay to suppress long-tail effect
    s_fid = math.exp(-fid / fid_tau)

    # ---------- 2. LPIPS (lower is better) ----------
    s_lpips = 1.0 - lpips
    s_lpips = max(0.0, min(1.0, s_lpips))  # safety clamp

    # ---------- 3. SSIM (higher is better) ----------
    s_ssim = max(0.0, min(1.0, ssim)) ** ssim_gamma

    # ---------- 4. PSNR (higher is better) ----------
    s_psnr = 1.0 / (1.0 + math.exp(-(psnr - psnr_mu) / psnr_sigma))

    # ---------- 5. Weighted aggregation ----------
    total_weight = w_fid + w_lpips + w_ssim + w_psnr
    assert abs(total_weight - 1.0) < 1e-6, "Metric weights must sum to 1."

    perceptual_score = (
        w_fid * s_fid +
        w_lpips * s_lpips +
        w_ssim * s_ssim +
        w_psnr * s_psnr
    )

    return 100.0 * perceptual_score


if __name__=="__main__":

    sora2_coin = {
        "name":"sora2_coin",
        "FVD":936.79,
        "LPIPS":0.282,
        "SSIM":0.340,
        "PSNR":13.473,
        "gpt":5.3287
    }

    sora2_droid = {
        "name":"sora2_droid",
        "FVD":2080.296,
        "LPIPS":0.390,
        "SSIM":0.189,
        "PSNR":10.490,
        "gpt":1.800
    }

    wan2_5_coin = {
        "name":"wan2_5_coin",
        "FVD":783.98,
        "LPIPS":0.253,
        "SSIM":0.382,
        "PSNR":14.339,
        "gpt":6.0280
    }

    wan2_5_droid = {
        "name":"wan2_5_droid",
        "FVD":1889.495,
        "LPIPS":0.411,
        "SSIM":0.160,
        "PSNR":9.810,
        "gpt":1.7133
    }

    wan2_2_coin = {
        "name":"wan2_2_coin",
        "FVD":1025.532,
        "LPIPS":0.305,
        "SSIM":0.312,
        "PSNR":12.025,
        "gpt":4.0699
    }

    wan2_2_droid = {
        "name":"wan2_2_droid",
        "FVD":2618.005,
        "LPIPS":0.428,
        "SSIM":0.153,
        "PSNR":9.226,
        "gpt":1.1467
    }

    gebase_coin = {
        "name":"gebase_coin",
        "FVD":3043.513,
        "LPIPS":0.531,
        "SSIM":0.139,
        "PSNR":8.600,
        "gpt":0.9028
    }

    gebase_droid = {
        "name":"gebase_droid",
        "FVD":2477.373,
        "LPIPS":0.417,
        "SSIM":0.169,
        "PSNR":9.592,
        "gpt":1.3033
    }

    sora2_crosstask = {
        "name":"sora2_crosstask",
        "FVD":1334.581,
        "LPIPS":0.273,
        "SSIM":0.293,
        "PSNR":13.273,
        "gpt":5.7632
    }

    sora2_fm = {
        "name":"spra2_fm",
        "FVD":1128.900,
        "LPIPS":0.392,
        "SSIM":0.240,
        "PSNR":11.067,
        "gpt":1.4787
    }

    wan2_5_crosstask = {
        "name":"wan2_5_crosstask",
        "FVD":985.221,
        "LPIPS":0.211,
        "SSIM":0.358,
        "PSNR":14.706,
        "gpt":6.8081
    }

    wan2_5_fm = {
        "name":"wan2_5_fm",
        "FVD":1756.251,
        "LPIPS":0.393,
        "SSIM":0.252,
        "PSNR":11.740,
        "gpt":1.4500
    }

    wan2_2_crosstask = {
        "name":"wan2_2_crosstask",
        "FVD":1354.888,
        "LPIPS":0.264,
        "SSIM":0.288,
        "PSNR":12.123,
        "gpt":5.610
    }

    wan2_2_fm = {
        "name":"wan2_2_fm",
        "FVD":2466.512,
        "LPIPS":0.438,
        "SSIM":0.197,
        "PSNR":9.846,
        "gpt":0.620
    }

    gebase_crosstask = {
        "name":"gebase_crosstask",
        "FVD":5364.688,
        "LPIPS":0.663,
        "SSIM":0.035,
        "PSNR":7.429,
        "gpt":0.204        
    }

    gebase_fm = {
        "name":"gebase_fm",
        "FVD":2151.393,
        "LPIPS":0.709,
        "SSIM":0.097,
        "PSNR":7.479,
        "gpt":0.0100        
    }

    jimeng_coin = {
        "name":"jimeng_coin",
        "FVD":1081.994,
        "LPIPS":0.317,
        "SSIM":0.265,
        "PSNR":12.549,
        "gpt":3.6638   
    }

    jimeng_droid = {
        "name":"jimeng_droid",
        "FVD":1480.175,
        "LPIPS":0.401,
        "SSIM":0.160,
        "PSNR":10.500,
        "gpt":1.3525  
    }

    seedance_coin = {
        "name":"seedance_coin",
        "FVD":1352.510,
        "LPIPS":0.305,
        "SSIM":0.257,
        "PSNR":12.495,
        "gpt":3.798          
    }

    seedance_droid = {
        "name":"seedance_droid",
        "FVD":2335.514,
        "LPIPS":0.413,
        "SSIM":0.121,
        "PSNR":9.710,
        "gpt":1.283          
    }


    gpt_image_1_atari = {
        "name":"gpt-image-1_atari",
        "FID":55.200,
        "LPIPS":0.382,
        "SSIM":0.515,
        "PSNR":13.832,
        "gpt":6.097
    }

    gpt_image_1_mw = {
        "name":"gpt-image-1_mw",
        "FID":64.912,
        "LPIPS":0.617,
        "SSIM":0.481,
        "PSNR":10.717,
        "gpt":3.254
    }

    qwen_image_edit_atari = {
        "name":"qwen_image_edit_atari",
        "FID":61.736,
        "LPIPS":0.280,
        "SSIM":0.718,
        "PSNR":17.041,
        "gpt":2.386
    }

    qwen_image_edit_mw = {
        "name":"qwen_image_edit_mw",
        "FID":46.177,
        "LPIPS":0.586,
        "SSIM":0.422,
        "PSNR":10.461,
        "gpt":2.894
    }

    nano_atari = {
        "name":"nano_atari",
        "FID":30.357,
        "LPIPS":0.181,
        "SSIM":0.650,
        "PSNR":17.313,
        "gpt":5.910
    }

    nano_mw = {
        "name":"nano_mw",
        "FID":47.279,
        "LPIPS":0.585,
        "SSIM":0.399,
        "PSNR":10.358,
        "gpt":3.374
    }

    diamond_atari = {
        "name":"diamond_atari",
        "FID":26.427,
        "LPIPS":0.109,
        "SSIM":0.735,
        "PSNR":21.753,
        "gpt":7.352       
    }

    diamond_mw = {
        "name":"diamond_mw",
        "FID":278.154,
        "LPIPS":0.961,
        "SSIM":0.3037,
        "PSNR":5.96,
        "gpt":0.391      
    }

    diamond_ac = {
        "name":"diamond_ac",
        "FID":220.656,
        "LPIPS":0.831,
        "SSIM":0.306,
        "PSNR":9.665,
        "gpt":0.99
    }

    gpt_image_1_ac = {
        "name":"gpt-image-1_ac",
        "FID":71.67,
        "LPIPS":0.628,
        "SSIM":0.394,
        "PSNR":11.518,
        "gpt": 3.905
    }

    nano_ac = {
        "name":"nano_ac",
        "FID":64.009,
        "LPIPS":0.651,
        "SSIM":0.347,
        "PSNR":9.705,
        "gpt": 3.407
    }

    qwen_image_edit_ac = {
        "name":"qwen_image_edit_ac",
        "FID":77.87,
        "LPIPS":0.662,
        "SSIM":0.386,
        "PSNR":10.93,
        "gpt":3.211
    }

    seedream_atari = {
        "name":"seedream_atari",
        "FID":74.21,
        "LPIPS":0.2861,
        "SSIM":0.4880,
        "PSNR":15.83,
        "gpt":5.247        
    }

    seedream_mw = {
        "name":"seedream_mw",
        "FID":127.481,
        "LPIPS":0.690,
        "SSIM":0.395,
        "PSNR":8.596,
        "gpt":2.602       
    }

    flux_atari = {
        "name":"flux_atari",
        "FID":49.99,
        "LPIPS":0.326,
        "SSIM":0.624,
        "PSNR":16.54,
        "gpt":3.753       
    }

    flux_mw = {
        "name":"flux_mw",
        "FID":31.792,
        "LPIPS":0.553,
        "SSIM":0.523,
        "PSNR":12.345,
        "gpt":3.467       
    }


    gpt_route_agent_coin = {
        "name":"gpt_route_agent_coin",
        "FVD":766.062,
        "LPIPS":0.303,
        "SSIM":0.492,
        "PSNR":14.900,
        "gpt":5.632    
    }

    gpt_route_agent_droid = {
        "name":"gpt_route_agent_droid",
        "FVD":1590.935,
        "LPIPS":0.407,
        "SSIM":0.243,
        "PSNR":9.761,
        "gpt":2.97    
    }

    gpt_route_agent_atari = {
        "name":"gpt_route_agent_atari",
        "FID":29.126,
        "LPIPS":0.127,
        "SSIM":0.717,
        "PSNR":21.924,
        "gpt":7.093    
    }

    gpt_route_agent_mw = {
        "name":"gpt_route_agent_mw",
        "FID":44.31,
        "LPIPS":0.5896,
        "SSIM":0.4608,
        "PSNR":11.75,
        "gpt":3.316  
    }

    qwen_route_agent_coin = {
        "name":"qwen_route_agent_coin",
        "FVD":747.618,
        "LPIPS":0.358,
        "SSIM":0.450,
        "PSNR":14.678,
        "gpt":6.458         
    }

    qwen_route_agent_droid = {
        "name":"qwen_route_agent_droid",
        "FVD":2024.755,
        "LPIPS":0.423,
        "SSIM":0.198,
        "PSNR":8.992,
        "gpt":1.687  
    }

    qwen_route_agent_atari = {
        "name":"qwen_route_agent_atari",
        "FID":29.528,
        "LPIPS":0.127,
        "SSIM":0.733,
        "PSNR":20.023,
        "gpt":7.173    
    }

    qwen_route_agent_mw = {
        "name":"qwen_route_agent_mw",
        "FID":51.760,
        "LPIPS":0.587,
        "SSIM":0.414,
        "PSNR":11.093,
        "gpt":3.213  
    }

    qwen_route_agent_crosstask = {
        "name":"qwen_route_agent_crosstask",
        "FVD":3823.0975,
        "LPIPS":0.372,
        "SSIM":0.112,
        "PSNR":10.879,
        "gpt":4.2727         
    }

    qwen_route_agent_fm = {
        "name":"qwen_route_agent_fm",
        "FVD":1907.485,
        "LPIPS":0.435,
        "SSIM":0.218,
        "PSNR":10.080,
        "gpt":2.1481         
    }

    qwen_route_agent_android = {
        "name":"qwen_route_agent_android",
        "FID":69.809,
        "LPIPS":0.611,
        "SSIM":0.390,
        "PSNR":11.630,
        "gpt":4.013    
    }    

    qwen_route_agent_new_tool_coin = {
        "name":"qwen_route_agent_new_tool_coin",
        "FVD":3881.533,
        "LPIPS":0.378,
        "SSIM":0.192,
        "PSNR":11.255,
        "gpt":3.153         
    }

    qwen_route_agent_new_tool_droid = {
        "name":"qwen_route_agent_new_tool_droid",
        "FVD":3228.235,
        "LPIPS":0.436,
        "SSIM":0.120,
        "PSNR":9.131,
        "gpt":1.1538        
    }

    qwen_route_agent_new_tool_atari = {
        "name":"qwen_route_agent_new_tool_atari",
        "FID":47.20,
        "LPIPS":0.099,
        "SSIM":0.563,
        "PSNR":20.584,
        "gpt":4.113    
    }

    qwen_route_agent_new_tool_mw = {
        "name":"qwen_route_agent_new_tool_mw",
        "FID":144.00,
        "LPIPS":0.706,
        "SSIM":0.395,
        "PSNR":7.496,
        "gpt":2.080  
    }

    qwen_origin_coin = {
        "name":"qwen_origin_coin",
        "FVD":4471.406,
        "LPIPS":0.3699,
        "SSIM":0.2250,
        "PSNR":12.332,
        "gpt":1.5
    }

    qwen_origin_droid = {
        "name":"qwen_origin_droid",
        "FVD":4542.212,
        "LPIPS":0.433,
        "SSIM":0.117,
        "PSNR":8.866,
        "gpt":1.8889
    }

    qwen_origin_atari = {
        "name":"qwen_origin_atari",
        "FID":79.469,
        "LPIPS":0.168,
        "SSIM":0.802,
        "PSNR":20.75,
        "gpt":6.746    
    }

    qwen_origin_mw = {
        "name":"qwen_origin_mw",
        "FID":88.959,
        "LPIPS":0.672,
        "SSIM":0.4285,
        "PSNR":9.637,
        "gpt":2.040    
    }    



    select_video_model = qwen_origin_droid

    uni_s = unified_video_alignment_score(
        fvd=select_video_model["FVD"], 
        lpips=select_video_model["LPIPS"], 
        ssim=select_video_model["SSIM"], 
        psnr=select_video_model["PSNR"]
    )
    print(f"unified_score for {select_video_model['name']} is")
    print(f"uni_s = {uni_s}")
    print(f"gpt_s = {select_video_model["gpt"]}")



    select_image_model = qwen_origin_mw

    uni_s = unified_image_alignment_score(
        fid=select_image_model["FID"], 
        lpips=select_image_model["LPIPS"], 
        ssim=select_image_model["SSIM"], 
        psnr=select_image_model["PSNR"]
    )
    print(f"unified_score for {select_image_model['name']} is")
    print(f"uni_s = {uni_s}")
    print(f"gpt_s = {select_image_model["gpt"]}")



