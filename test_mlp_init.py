"""
test_mlp_init.py

Script de diagnóstico para verificar que el MLP se inicializa correctamente
y que los fixes resolvieron el problema de logits ≈ 0.

EJECUCIÓN:
    python test_mlp_init.py

SALIDA ESPERADA DESPUÉS DE FIX:
    ✓ fc1.weight mean abs: 0.07-0.12  (NO < 0.001)
    ✓ Logits mean abs: 0.5-2.0        (NO < 0.01)
    ✓ Logits std: 0.5-1.5             (NO < 0.1)
    ✓ Test accuracy before: ~10%
    ✓ Test accuracy after SGD: >50%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from Model.mlp_pytorch import MLPPyTorch


def test_initialization():
    """Test: Verificar que la inicialización es patológicum"""
    print("=" * 70)
    print("TEST 1: INICIALIZACIÓN DEL MLP")
    print("=" * 70)
    
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)
    
    # Verificar pesos fc1
    fc1_mean = mlp.fc1.weight.data.abs().mean().item()
    fc1_std = mlp.fc1.weight.data.std().item()
    
    print(f"\nfc1.weight:")
    print(f"  - mean abs: {fc1_mean:.6f}")
    print(f"  - std:      {fc1_std:.6f}")
    
    if fc1_mean < 0.001:
        print(f"  ✗ [FALLO] fc1_mean < 0.001 → logits será ≈ 0")
        return False
    else:
        print(f"  ✓ [OK] fc1_mean > 0.001")
    
    # Verificar que no es muy grande tampoco
    if fc1_mean > 1.0:
        print(f"  ⚠  [AVISO] fc1_mean > 1.0 → puede causar inestabilidad")
    
    return True


def test_forward_pass():
    """Test: Verificar logits en forward pass"""
    print("\n" + "=" * 70)
    print("TEST 2: FORWARD PASS CON FEATURES NORMALIZADAS")
    print("=" * 70)
    
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)
    mlp.eval()
    
    # Simular features desde CNN (ImageNet normalizadas)
    batch_size = 32
    features = torch.randn(batch_size, 512) * 0.5  # Rango [-1, 1] típico
    
    with torch.no_grad():
        logits = mlp(features)
    
    logits_mean = logits.mean().item()
    logits_std = logits.std().item()
    logits_max = logits.max().item()
    logits_min = logits.min().item()
    
    print(f"\nLogits stats (batch_size={batch_size}):")
    print(f"  - mean:     {logits_mean:.6f}")
    print(f"  - std:      {logits_std:.6f}")
    print(f"  - min:      {logits_min:.6f}")
    print(f"  - max:      {logits_max:.6f}")
    
    if abs(logits_mean) < 0.01 and logits_std < 0.1:
        print(f"  ✗ [FALLO] Logits patológicamente pequeños")
        print(f"      → softmax ≈ uniforme → accuracy ≈ 0%")
        return False
    else:
        print(f"  ✓ [OK] Logits tienen varianza razonable")
    
    # Verificar softmax
    probs = torch.softmax(logits, dim=1)
    max_prob_mean = probs.max(dim=1)[0].mean().item()
    min_prob_mean = probs.min(dim=1)[0].mean().item()
    
    print(f"\nSoftmax stats:")
    print(f"  - max prob (avg): {max_prob_mean:.4f}")
    print(f"  - min prob (avg): {min_prob_mean:.6f}")
    print(f"  - uniform prob:   0.001 (1/1000)")
    
    if max_prob_mean < 0.002:
        print(f"  ✗ [FALLO] Probabilidades casi uniformes → sin información")
        return False
    else:
        print(f"  ✓ [OK] Distribución tiene estructura")
    
    return True


def test_training_step():
    """Test: Un paso de SGD debe cambiar los parámetros significativamente"""
    print("\n" + "=" * 70)
    print("TEST 3: ENTRENAMIENTO (GRADIENT DESCENT)")
    print("=" * 70)
    
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)
    mlp.train()
    
    # Datos simulados
    batch_size = 32
    features = torch.randn(batch_size, 512)
    targets = torch.randint(0, 1000, (batch_size,))
    
    # Forward before
    with torch.no_grad():
        logits_before = mlp(features).clone()
        accuracy_before = (logits_before.argmax(1) == targets).sum().item() / batch_size
    
    print(f"\nAntes de SGD:")
    print(f"  - accuracy: {accuracy_before*100:.2f}%")
    print(f"  - logits[0,0:5]: {logits_before[0, :5]}")
    
    # Training step
    mlp.zero_grad()
    logits = mlp(features)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    
    with torch.no_grad():
        for p in mlp.parameters():
            if p.grad is not None:
                p.data -= 0.01 * p.grad
    
    # Forward after
    mlp.eval()
    with torch.no_grad():
        logits_after = mlp(features)
        accuracy_after = (logits_after.argmax(1) == targets).sum().item() / batch_size
    
    print(f"\nDespués de SGD (1 step, lr=0.01):")
    print(f"  - accuracy: {accuracy_after*100:.2f}%")
    print(f"  - logits[0,0:5]: {logits_after[0, :5]}")
    print(f"  - loss: {loss.item():.4f}")
    
    # Verificar cambio
    param_change = (logits_after - logits_before).abs().mean().item()
    print(f"  - cambio en logits: {param_change:.6f}")
    
    if param_change < 0.001:
        print(f"  ✗ [FALLO] Parámetros NO cambian → no hay learning")
        return False
    else:
        print(f"  ✓ [OK] Parámetros se actualizan")
    
    return True


def main():
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO: MLP INITIALIZATION FIX")
    print("=" * 70 + "\n")
    
    tests = [
        ("Inicialización", test_initialization),
        ("Forward Pass", test_forward_pass),
        ("Entrenamiento", test_training_step),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} | {name}")
    
    print("\n" + "=" * 70)
    all_pass = all(r for _, r in results)
    if all_pass:
        print("✓ TODOS LOS TESTS PASARON - MLP ESTÁ BIEN CONFIGURADO")
        print("\nAHORA EJECUTA EL ENTRENAMIENTO CON:")
        print("  Terminal 1: python ps_imagenet.py --cnn-arch simple")
        print("  Terminal 2: python worker_imagenet.py --device cpu")
        print("\nDEBERÍAS VER:")
        print("  - loss comenzando alto (≈6.9) pero BAJANDO")
        print("  - acc comenzando ≈0% pero SUBIENDO")
        print("  - Cambios visibles en cada reporte")
        return 0
    else:
        print("✗ ALGUNOS TESTS FALLARON - VERIFICA LOS LOGS ARRIBA")
        return 1


if __name__ == "__main__":
    sys.exit(main())
