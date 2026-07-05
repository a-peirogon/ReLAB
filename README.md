# ReLAB
Gymnasium-based environment for RL research/experiments

<table>
  <tr>
    <td align="center"><img src="./media/snake.gif" width="380"/></td>
    <td align="center"><img src="./media/robot_nav.gif" width="380"/></td>
  </tr>
  <tr>
    <td align="center"><b>Snake</b></td>
    <td align="center"><b>Robot navigation</b></td>
  </tr>
</table>

## Experimentos

**Snake** — observación vectorial pura (11 floats: peligro adelante/derecha/izquierda, dirección actual, posición relativa de la comida). Agentes utilizados: Q-learning tabular, DQN.

**Robot_nav** — robot de tracción diferencial debe alcanzar una meta evadiendo obstáculos móviles en un mundo 2D continuo. Incluye un DQN entrenable, también DWA (*Dynamic Window Approach*).
