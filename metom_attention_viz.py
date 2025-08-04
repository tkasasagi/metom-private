import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoModel, AutoProcessor
import seaborn as sns

def inspect_model_structure(model):
    """モデル構造を詳細に調査"""
    print("=== Model Structure Investigation ===")
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    print("\n=== Named Modules ===")
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower() or 'transformer' in name.lower():
            print(f"{name}: {type(module)}")
    
    print("\n=== Model Attributes ===")
    for attr in dir(model):
        if not attr.startswith('_') and hasattr(model, attr):
            try:
                value = getattr(model, attr)
                if hasattr(value, '__class__') and 'torch' in str(value.__class__):
                    print(f"{attr}: {type(value)}")
            except:
                pass
    
    print("\n=== Forward Method Signature ===")
    import inspect
    try:
        sig = inspect.signature(model.forward)
        print(f"forward{sig}")
    except:
        print("Could not inspect forward method")

def inspect_metom_attention(model):
    """MetomAttentionの詳細な構造調査"""
    print("=== MetomAttention Structure Investigation ===")
    
    attention_module = model.transformer.layers[0][0]  # 最初の層のattention
    print(f"Attention module type: {type(attention_module)}")
    
    print("\n=== Attention Module Attributes ===")
    for attr in dir(attention_module):
        if not attr.startswith('_'):
            try:
                value = getattr(attention_module, attr)
                if not callable(value):
                    print(f"{attr}: {value} (type: {type(value)})")
            except:
                pass
    
    print("\n=== to_qkv Linear Layer Info ===")
    qkv_layer = attention_module.to_qkv
    print(f"to_qkv input features: {qkv_layer.in_features}")
    print(f"to_qkv output features: {qkv_layer.out_features}")
    
    # heads数とdim_headを推定
    hidden_dim = qkv_layer.in_features
    qkv_dim = qkv_layer.out_features
    
    print(f"Hidden dimension: {hidden_dim}")
    print(f"QKV total dimension: {qkv_dim}")
    print(f"QKV dimension should be 3 * heads * dim_head = {qkv_dim}")
    
    # 一般的なhead数で試す
    possible_heads = [1, 2, 4, 8, 12, 16]
    for heads in possible_heads:
        if (qkv_dim // 3) % heads == 0:
            dim_head = (qkv_dim // 3) // heads
            print(f"Possible: heads={heads}, dim_head={dim_head}")
    
    return attention_module

class MetomAttentionVisualizer:
    def __init__(self, model_name="SakanaAI/Metom", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            _attn_implementation="eager",
            trust_remote_code=True
        ).to(device=device)
        
        # モデル構造を調査
        inspect_model_structure(self.model)
        
        # MetomAttentionの詳細調査
        self.attention_info = inspect_metom_attention(self.model)
        
        # Attention mapを保存するための変数
        self.attention_maps = []
        self.hooks = []
        
    def register_metom_attention_hooks(self):
        """MetomAttention専用のhook"""
        def metom_attention_hook(module, input, output):
            module_name = None
            for name, mod in self.model.named_modules():
                if mod is module:
                    module_name = name
                    break
            
            print(f"MetomAttention hook triggered: {module_name}")
            print(f"Input type: {type(input)}, length: {len(input) if isinstance(input, tuple) else 'not tuple'}")
            print(f"Output type: {type(output)}")
            
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
                print(f"Input tensor shape: {x.shape}")
                
                # MetomAttentionの内部処理を手動で実行してattention weightを取得
                try:
                    # to_qkv で Query, Key, Value を取得
                    qkv = module.to_qkv(module.norm(x))
                    print(f"QKV shape: {qkv.shape}")
                    
                    # reshape to separate q, k, v
                    b, n, _ = qkv.shape
                    qkv = qkv.reshape(b, n, 3, module.heads, module.dim_head)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, n, dim_head)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    print(f"Q shape: {q.shape}, K shape: {k.shape}")
                    
                    # attention weights を計算
                    dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
                    attn = module.attend(dots)  # softmax
                    
                    print(f"Attention weights shape: {attn.shape}")
                    self.attention_maps.append(attn.detach().cpu())
                    
                except Exception as e:
                    print(f"Error in manual attention calculation: {e}")
        
        # 全てのMetomAttentionモジュールにhookを登録
        for name, module in self.model.named_modules():
            if isinstance(module, type(self.model.transformer.layers[0][0])):  # MetomAttention
                print(f"Registering MetomAttention hook for: {name}")
                hook = module.register_forward_hook(metom_attention_hook)
                self.hooks.append(hook)
    
    def manual_attention_extraction(self, image_array, layer_idx=-1):
        """手動でattention weightを抽出 (MetomAttention対応版)"""
        print(f"=== Manual attention extraction for layer {layer_idx} ===")
        
        with torch.no_grad():
            # パッチ埋め込み
            patches = self.model.to_patch_embedding(image_array)
            b, n, _ = patches.shape
            
            # CLSトークンを追加
            cls_tokens = self.model.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, patches), dim=1)
            x += self.model.pos_embedding[:, :(n + 1)]
            x = self.model.dropout(x)
            
            print(f"Input to transformer: {x.shape}")
            
            # 指定された層まで進む
            target_layer = layer_idx if layer_idx >= 0 else len(self.model.transformer.layers) + layer_idx
            print(f"Target layer: {target_layer}")
            
            for i, layer in enumerate(self.model.transformer.layers):
                if i == target_layer:
                    # この層のattentionを手動計算
                    attention_module = layer[0]  # MetomAttention
                    
                    print(f"Processing layer {i}")
                    
                    # Attention計算
                    normed_x = attention_module.norm(x)
                    qkv = attention_module.to_qkv(normed_x)
                    
                    print(f"QKV shape: {qkv.shape}")
                    
                    # QKVの次元を推定
                    b, n, qkv_dim = qkv.shape
                    hidden_dim = normed_x.shape[-1]
                    
                    # heads数を推定 (一般的な値を試す)
                    possible_heads = [1, 2, 4, 8, 12, 16]
                    heads = None
                    dim_head = None
                    
                    for h in possible_heads:
                        if (qkv_dim // 3) % h == 0:
                            potential_dim_head = (qkv_dim // 3) // h
                            if h * potential_dim_head * 3 == qkv_dim:
                                heads = h
                                dim_head = potential_dim_head
                                break
                    
                    if heads is None:
                        # フォールバック: 最も可能性の高い値
                        heads = 8  # 一般的な値
                        dim_head = (qkv_dim // 3) // heads
                        print(f"Warning: Using fallback heads={heads}, dim_head={dim_head}")
                    
                    print(f"Inferred: heads={heads}, dim_head={dim_head}")
                    
                    # QKVを分割
                    qkv = qkv.reshape(b, n, 3, heads, dim_head)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, n, dim_head)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    print(f"Q shape: {q.shape}, K shape: {k.shape}")
                    
                    # Attention weights
                    scale = dim_head ** -0.5  # スケールファクター
                    dots = torch.matmul(q, k.transpose(-1, -2)) * scale
                    attn_weights = attention_module.attend(dots)  # softmax
                    
                    print(f"Layer {i} attention weights shape: {attn_weights.shape}")
                    return attn_weights.cpu()
                
                # 層を通す
                x = layer[0](x) + x  # attention + residual
                x = layer[1](x) + x  # feedforward + residual
                
            print(f"Target layer {target_layer} not found in {len(self.model.transformer.layers)} layers")
            return None
    
    def remove_hooks(self):
        """登録したhookを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_image(self, image_url):
        """URLから画像を取得"""
        if image_url.startswith('http'):
            return Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        else:
            return Image.open(image_url).convert("RGB")
    
    def visualize_attention(self, image_path, layer_idx=-1, head_idx=0, save_path=None):
        """
        Attention mapを可視化 (Metom専用改良版)
        """
        # 画像を読み込み
        image = self.get_image(image_path)
        image_array = self.processor(images=image, return_tensors="pt")["pixel_values"]
        image_array = image_array.to(device=self.device, dtype=torch.float32)
        
        print(f"Input image shape: {image_array.shape}")
        
        # Method 1: 手動でattention weightを抽出
        print("Attempting manual attention extraction...")
        attention_weights = self.manual_attention_extraction(image_array, layer_idx)
        
        if attention_weights is not None:
            print("Successfully obtained attention weights via manual extraction!")
            predictions = self.model.get_predictions(image_array)
            return self._visualize_from_weights(image, attention_weights, layer_idx, head_idx, save_path, predictions[0])
        
        # Method 2: MetomAttention専用hook
        print("Trying MetomAttention hook approach...")
        self.attention_maps = []
        self.register_metom_attention_hooks()
        
        with torch.inference_mode():
            predictions = self.model.get_predictions(image_array)
            print(f"Predictions: {predictions}")
            
        self.remove_hooks()
        
        if self.attention_maps:
            print(f"Captured {len(self.attention_maps)} attention maps via hooks")
            attention_weights = self.attention_maps[layer_idx] if layer_idx < len(self.attention_maps) else self.attention_maps[-1]
            return self._visualize_from_weights(image, attention_weights, layer_idx, head_idx, save_path, predictions[0])
        
        print("Could not obtain attention weights through any method.")
        return None, None
    
    def _visualize_from_weights(self, image, attention_weights, layer_idx, head_idx, save_path, prediction="Unknown"):
        """Attention weightsから可視化を作成"""
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # 適切な次元を選択
        if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            attention = attention_weights[0, head_idx]
        elif attention_weights.dim() == 3:  # [heads, seq_len, seq_len]
            attention = attention_weights[head_idx]
        else:
            attention = attention_weights
        
        print(f"Selected attention shape: {attention.shape}")
        
        # CLSトークンから各パッチへの注意度 (0番目がCLSトークン)
        if attention.shape[0] > 1:
            cls_attention = attention[0, 1:]  # CLSトークン以外への注意
        else:
            # 全体の平均を取る場合
            cls_attention = attention.mean(dim=0)
        
        print(f"CLS attention shape: {cls_attention.shape}")
        
        # パッチ数から画像サイズを推定 (128x128画像, 16x16パッチなら8x8=64パッチ)
        num_patches = len(cls_attention)
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            print(f"Warning: Cannot form perfect square. Patches: {num_patches}")
            # 最も近い正方形を使用
            patch_h = int(np.sqrt(num_patches))
            patch_w = (num_patches + patch_h - 1) // patch_h
            # 余りがある場合はパディング
            if patch_h * patch_w > num_patches:
                padding = patch_h * patch_w - num_patches
                cls_attention = torch.cat([cls_attention, torch.zeros(padding)])
        else:
            patch_h = patch_w = patch_size
        
        attention_map = cls_attention.reshape(patch_h, patch_w)
        
        # 可視化
        self._create_visualization(image, attention_map.numpy(), save_path, 
                                 title=f"Attention Map (Layer {layer_idx}, Head {head_idx})",
                                 prediction=prediction)
        
        return attention_map, prediction
    
    def _gradcam_alternative(self, image, image_array, save_path):
        """GradCAMライクな代替手法"""
        print("Using GradCAM-like alternative...")
        
        # 勾配ベースの可視化
        image_array.requires_grad_(True)
        
        with torch.enable_grad():
            outputs = self.model(image_array)
            
            # 最も可能性の高いクラスのスコア
            if hasattr(outputs, 'logits'):
                target_score = outputs.logits.max()
            else:
                # 予測関数を使って最高スコアを取得
                predictions = self.model.get_predictions(image_array)
                # モデルの内部でlogitsを取得する方法を探す
                target_score = torch.tensor(1.0, requires_grad=True)  # プレースホルダー
        
        if image_array.grad is not None:
            gradients = image_array.grad.data.abs().mean(dim=1)  # チャンネル方向で平均
            gradients = gradients.squeeze()
            
            # 可視化
            self._create_visualization(image, gradients.cpu().numpy(), save_path, title="Gradient-based Attention")
            return gradients.cpu().numpy(), None
        
        print("Could not generate any attention visualization.")
        return None
    
    def _create_visualization(self, image, attention_map, save_path, title="Attention Map", prediction="Unknown"):
        """可視化を作成"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元画像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_map, cmap='Blues', interpolation='nearest')
        axes[1].set_title(title)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 重ね合わせ
        # attention_mapを正規化
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        attention_resized = np.array(Image.fromarray(
            (attention_normalized * 255).astype(np.uint8)
        ).resize(image.size))
        
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='Blues', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'Prediction: {prediction}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_multiple_heads(self, image_path, layer_idx=-1, num_heads=4):
        """複数のAttention headを比較"""
        image = self.get_image(image_path)
        image_array = self.processor(images=image, return_tensors="pt")["pixel_values"]
        image_array = image_array.to(device=self.device, dtype=torch.float32)
        
        self.attention_maps = []
        self.register_attention_hooks()
        
        with torch.inference_mode():
            predictions = self.model.get_predictions(image_array)
            
        self.remove_hooks()
        
        if not self.attention_maps:
            return None
            
        attention = self.attention_maps[layer_idx]
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if num_heads > 2 else [axes] if num_heads == 1 else axes
        
        for head_idx in range(min(num_heads, attention.shape[1])):
            if attention.dim() == 4:
                head_attention = attention[0, head_idx]
            else:
                head_attention = attention[head_idx]
                
            cls_attention = head_attention[0, 1:]
            num_patches = int(np.sqrt(len(cls_attention)))
            attention_map = cls_attention.reshape(num_patches, num_patches)
            
            im = axes[head_idx].imshow(attention_map, cmap='hot', interpolation='nearest')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].axis('off')
            plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Multiple Attention Heads - Prediction: {predictions[0]}', fontsize=16)
        plt.tight_layout()
        plt.show()

# 使用例
if __name__ == "__main__":
    # 可視化クラスを初期化
    visualizer = MetomAttentionVisualizer()
    
    # サンプル画像で可視化
    sample_image = "/home/tarinc_sakana_ai/Metom/examples/example3_5009.jpg"
    
    # Attention mapを可視化
    try:
        attention_map, prediction = visualizer.visualize_attention(
            sample_image, 
            layer_idx=-1,  # 最後の層
            head_idx=0,    # 最初のhead
            save_path="attention_visualization.png"
        )
        
        if attention_map is not None:
            print(f"Successfully visualized attention for prediction: {prediction}")
        else:
            print("Failed to generate attention visualization")
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()