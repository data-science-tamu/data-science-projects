# Documentation for Pytorch Models

## Provided Data
**disp_coord**
- Discrete coordinates [0, 256-0] - [256, 256-0]

**disp_data**
- A bunch of float pairs (what does it represent?). 
- *Is it the displacement at each of the discrete coordinates in disp_coord?*
- *Based on it's usage in the provided code, seems to represent the axial displacement.*

**m_data**
- Young's Modulus? $256^2$ data points.
- Is this representing the $E$?

**nu_data**
- Poisson's Ration, $\nu$.
- $256^2$ data points.

**strain_coord**
- $256^2$ data points
- Discretized coordinates? [0.5, 255.5-0.5] - [255.5, 255.5-0.5]
- Likely the discretized coordinates for **strain data**

**strain_data**
- $256^2$ data points
- A bunch of float triplets. Could they represent:
  - $\epsilon_{xx}$
  - $\epsilon_{yy}$
  - $\gamma_{xy}$
- Which would then mean it represents the displacements
  - So would be used when positions are known, but both elasticity values (young's modulus and poisson's ratio) are unknown.



## The Inverse Problem
### Source
#### Section 2.4 of paper

>In this section, we examine inverse elasticity problems in which the 
displacement field of a given object is known and the objective is to determine 
the elasticity field. In ElastNet-2021, we assumed that the material was 
incompressible. In this work, we remove the assumption and use two elastic 
constants to describe the elasticity of the material. To illustrate this, a 
model with a dragon-shaped hard inclusion is considered, as shown in Figure 4a. 
The Poisson's ratio is determined by a rooster-shaped pattern. In this 
“dragon & rooster” model, the Young's modulus varies from 0.1 to 1.0 MPa. 
The Poisson's ratio varies from 0.1 to 0.5, as most materials have a Poisson's 
ratio in this range. Although some materials may have a zero Poisson's ratio, 
they are not considered because the MRE diverges when Poisson's ratio 
approaches zero. An average normal strain (εxx) of 1% is introduced by the 
displacements applied along the x-direction on the boundary. The displacement 
and strain fields are shown in Figure 4a,b, respectively. In this inverse 
elasticity problem, the unknown physical quantities are the Young's modulus 
and Poisson's ratio fields. The predictions and relative error maps are shown 
in Figure 4c, and the learning process is shown in Movie S2, Supporting 
Information. We demonstrate that ElastNet can simultaneously reconstruct these 
two elastic constants with high accuracy, with the MRE 0.58% for Young's 
modulus and 1.31% for Poisson's ratio. 