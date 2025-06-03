---
library_name: stable-baselines3
tags:
- LunarLander-v3
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: LunarLander-v3
      type: LunarLander-v3
    metrics:
    - type: mean_reward
      value: 280.77 +/- 18.31
      name: mean_reward
      verified: false
---

# PPO-LunarLander-v3

Este repositorio contiene un modelo entrenado con el algoritmo **Proximal Policy Optimization (PPO)** para el entorno **LunarLander-v3** de OpenAI Gym. El modelo fue entrenado utilizando la biblioteca Stable-Baselines3 y logra un rendimiento sólido al aterrizar un módulo lunar de manera segura.

## 📖 Descripción del Proyecto

El objetivo de este proyecto es entrenar un agente de aprendizaje por refuerzo (RL) para resolver el entorno **LunarLander-v3**, donde el agente debe controlar un módulo lunar para aterrizar de forma segura en una plataforma designada. El agente aprende a equilibrar la eficiencia del combustible, la estabilidad y la precisión en el aterrizaje mediante prueba y error, optimizando sus acciones con el algoritmo PPO.

### Detalles del Entorno
- **Entorno**: LunarLander-v3 (OpenAI Gym)
- **Tarea**: Controlar un módulo lunar para aterrizar en una plataforma objetivo ajustando el empuje y la orientación.
- **Espacio de Observación**: Espacio continuo de 8 dimensiones (posición, velocidad, ángulo, velocidad angular, etc.).
- **Espacio de Acciones**: 4 acciones discretas (no hacer nada, encender motor izquierdo, encender motor principal, encender motor derecho).
- **Recompensa**: Positiva por aterrizajes exitosos, negativa por choques o uso excesivo de combustible.

### Arquitectura del Modelo
- **Algoritmo**: Proximal Policy Optimization (PPO)
- **Biblioteca**: Stable-Baselines3
- **Detalles del Entrenamiento**:
  - Pasos totales: 1,507,328
  - Tasa de aprendizaje: 0.0003
  - Rango de recorte: 0.2
  - Iteraciones de entrenamiento: 46
  - Recompensa media final por episodio: 284
  - Longitud media final de episodios: 215 pasos

### Resultados del Entrenamiento
Los registros de entrenamiento muestran una mejora constante en el rendimiento:
- **Recompensa Media por Episodio**: Aumentó de ~132 a **284** en 46 iteraciones.
- **Longitud Media de Episodios**: Se estabilizó en alrededor de 215 pasos, indicando aterrizajes eficientes.
- **Varianza Explicada**: Alcanzó 0.997, mostrando que la función de valor predice con precisión los retornos.
- **Pérdida de Gradiente de Política**: Convergió a valores negativos pequeños, indicando actualizaciones estables de la política.
- **Pérdida de Entropía**: Disminuyó a -0.711, reflejando una política más segura con el tiempo.

### Evaluación
El modelo entrenado fue evaluado en 10 episodios:
- **Recompensa Media**: 272.80 ± 18.08
- Esto indica un rendimiento consistente y robusto, con el agente logrando aterrizajes exitosos en la mayoría de los episodios.

## 📹 Demostración
Un video del agente entrenado actuando en el entorno LunarLander-v3 está disponible en el repositorio (`./videos/rl-video-step-0-to-step-250.mp4`). El agente demuestra un control suave y un comportamiento de aterrizaje preciso.

## 🚀 Primeros Pasos

### Requisitos Previos
Para usar este modelo, asegúrate de tener instaladas las siguientes dependencias:
```bash
pip install stable-baselines3 gym-box2d huggingface_sb3
```

### Cargar el Modelo
Puedes cargar y usar el modelo desde este repositorio con el siguiente código:

```python
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
import gym

# Cargar el modelo
repo_id = "IntelliGrow/PPO-LunarLander-v3"
model_filename = "PPO-LunarLander-v3.zip"
model = PPO.load(load_from_hub(repo_id, model_filename))

# Crear el entorno
env = gym.make("LunarLander-v3", render_mode="human")

# Ejecutar el modelo
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

### Grabar un Video
Para grabar un video del rendimiento del agente, usa `VecVideoRecorder` de Stable-Baselines3:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym
import os

# Crear entorno
env_id = "LunarLander-v3"
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)
eval_env = gym.make(env_id, render_mode="rgb_array")
eval_env = DummyVecEnv([lambda: eval_env])

# Configurar el grabador de video
video_env = VecVideoRecorder(
    eval_env,
    video_folder,
    record_video_trigger=lambda step: step == 0,
    video_length=250
)

# Ejecutar y grabar
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
El modelo fue subido al Hugging Face Hub usando la función `package_to_hub` de `huggingface_sb3`. El repositorio incluye:
- Pesos del modelo entrenado (`PPO-LunarLander-v3.zip`)
- Estados de la política y el optimizador (`policy.pth`, `policy.optimizer.pth`, `pytorch_variables.pth`)
- Un video que muestra el rendimiento del agente

Para subir tu propio modelo al Hugging Face Hub, asegúrate de tener un token válido de Hugging Face y usa lo siguiente:

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
Siéntete libre de hacer un fork de este repositorio, experimentar con el modelo o ajustarlo aún más. Si encuentras problemas o tienes sugerencias, por favor abre un issue en [huggingface_sb3](https://github.com/huggingface/huggingface_sb3).

## 📜 Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) por la implementación de PPO.
- [Hugging Face Hub](https://huggingface.co/) por alojar el modelo.
- [OpenAI Gym](https://gym.openai.com/) por el entorno LunarLander-v3.

---

**Autor**: IntelliGrow  
**Repositorio**: [IntelliGrow/PPO-LunarLander-v3](https://huggingface.co/IntelliGrow/PPO-LunarLander-v3)  
**Mensaje de Commit**: Subida de Algoritmo LunarLanderV3 by me
```