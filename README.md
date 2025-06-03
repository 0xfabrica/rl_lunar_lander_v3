# PPO-LunarLander-v3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-%3E=1.6.0-005571?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjZmZmIiBoZWlnaHQ9IjEyIiB2aWV3Qm94PSIwIDAgMTIgMTIiIHdpZHRoPSIxMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNNSAxMUw2IDExIDYgOCA1IDhNNiAxMVY4bTAgMEw2IDVtMCAwbDAtM20wIDB2LTMuNSIgLz48L3N2Zz4=)
![Gym](https://img.shields.io/badge/Gym-box2d-green?logo=OpenAI)
![Hugging Face](https://img.shields.io/badge/Hugging--Face-Model-yellow?logo=huggingface)

Este repositorio contiene un modelo entrenado con el algoritmo **Proximal Policy Optimization (PPO)** para el entorno **LunarLander-v3** de OpenAI Gym. El modelo fue entrenado utilizando la biblioteca Stable-Baselines3 y está disponible para descarga y evaluación.

## 📖 Descripción del Proyecto

El objetivo de este proyecto es entrenar un agente de aprendizaje por refuerzo (RL) para resolver el entorno **LunarLander-v3**, donde el agente debe controlar un módulo lunar para aterrizar de forma segura en una plataforma objetivo.

### Detalles del Entorno

- **Entorno:** LunarLander-v3 (OpenAI Gym)
- **Tarea:** Controlar un módulo lunar para aterrizar en una plataforma ajustando el empuje y la orientación
- **Observaciones:** 8 dimensiones (posición, velocidad, ángulo, etc.)
- **Acciones:** 4 discretas (nada, motor izquierdo, motor principal, motor derecho)
- **Recompensa:** Positiva por aterrizajes exitosos, negativa por choques o uso excesivo de combustible

### Arquitectura del Modelo

- **Algoritmo:** Proximal Policy Optimization (PPO)
- **Biblioteca:** Stable-Baselines3
- **Entrenamiento:**
  - Pasos totales: 1,507,328
  - Tasa de aprendizaje: 0.0003
  - Rango de recorte: 0.2
  - Iteraciones: 46
  - Recompensa media final: 284
  - Longitud media de episodios: 215 pasos

### Resultados del Entrenamiento

- **Recompensa media por episodio:** de ~132 a **284** en 46 iteraciones
- **Longitud media:** ~215 pasos
- **Varianza explicada:** 0.997
- **Pérdida de gradiente de política:** convergió a valores pequeños
- **Pérdida de entropía:** descendió a -0.711

### Evaluación

- **Recompensa media en evaluación:** 272.80 ± 18.08 (10 episodios)
- El agente logra aterrizajes exitosos de manera consistente.

## 📹 Demostración

Un video mostrando el agente entrenado actuando en el entorno está disponible en `./videos/rl-video-step-0-to-step-250.mp4`.

## 🚀 Primeros Pasos

### Requisitos Previos

```bash
pip install stable-baselines3 gym-box2d huggingface_sb3
```

### Cargar el Modelo

```python
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
import gym

repo_id = "IntelliGrow/PPO-LunarLander-v3"
model_filename = "PPO-LunarLander-v3.zip"
model = PPO.load(load_from_hub(repo_id, model_filename))

env = gym.make("LunarLander-v3", render_mode="human")
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

### Grabar un Video

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym
import os

env_id = "LunarLander-v3"
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)
eval_env = gym.make(env_id, render_mode="rgb_array")
eval_env = DummyVecEnv([lambda: eval_env])

video_env = VecVideoRecorder(
    eval_env,
    video_folder,
    record_video_trigger=lambda step: step == 0,
    video_length=250
)

obs = video_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = video_env.step(action)
    done = dones[0]
video_env.close()
print(f"Video guardado en {video_folder}")
```

## 📤 Subida a Hugging Face

El modelo fue subido al Hugging Face Hub usando `package_to_hub` de `huggingface_sb3`. El repositorio incluye pesos, estados y videos de desempeño.

```python
from huggingface_sb3 import package_to_hub

package_to_hub(
    model=model,
    model_name="PPO-LunarLander-v3",
    model_architecture="PPO",
    env_id="LunarLander-v3",
    eval_env=DummyVecEnv([lambda: gym.make("LunarLander-v3", render_mode="rgb_array")]),
    repo_id="TuUsuario/PPO-LunarLander-v3",
    commit_message="Subida del modelo PPO entrenado para LunarLander-v3"
)
```

## 🙌 Contribuciones

¡Haz fork, experimenta y ajusta el modelo! Si encuentras problemas o tienes sugerencias, abre un issue.

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Hugging Face Hub](https://huggingface.co/)
- [OpenAI Gym](https://gym.openai.com/)

---

**Autor:** IntelliGrow  
**Repositorio:** [IntelliGrow/PPO-LunarLander-v3](https://huggingface.co/IntelliGrow/PPO-LunarLander-v3)  
**Mensaje de Commit:** Subida de Algoritmo LunarLanderV3 by me

---

¿Quieres que lo guarde directamente en tu repositorio o deseas algún cambio adicional?
