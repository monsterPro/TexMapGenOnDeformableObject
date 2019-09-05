/**
 * From the OpenGL Programming wikibook: http://en.wikibooks.org/wiki/OpenGL_Programming
 * This file is in the public domain.
 * Contributors: Martin Kraus, Sylvain Beucler
 */
attribute vec4 v_coord;
attribute vec3 v_normal;
attribute vec2 v_tex_coord;
attribute vec3 v_img_coord;
varying vec4 position;  // position of the vertex (and fragment) in world space
varying vec3 varyingNormalDirection;  // surface normal vector in world space
uniform mat4 m, v, p;
uniform mat3 m_3x3_inv_transp;
varying vec2 tex_coord;
varying vec3 img_tex_coord;

void main()
{
  position = m * v_coord;
  varyingNormalDirection = normalize(m_3x3_inv_transp * v_normal);

  mat4 mvp = p*v*m;
  gl_Position = mvp * v_coord;

  vec4 sPlane = 1*vec4(1.0, 0.0, 0.0, 0.0);
  vec4 tPlane = 1*vec4(0.0, 1.0, 0.0, 0.0);

  //tex_coord = vec2(gl_MultiTexCoord0);
  tex_coord = vec2(v_tex_coord.x, -v_tex_coord.y);
  img_tex_coord = v_img_coord;
  //tex_coord = 0.01 * gl_Vertex.xz;

  // ---- Object linear
  //tex_coord.x = dot(position + vec4(0.5, 0, 0, 0), sPlane);
  //tex_coord.y = dot(position + vec4(0, 1.0, 0, 0), tPlane);

  // ---- Sphere Map
  //vec3 u = normalize( vec3(gl_ModelViewMatrix * position) );
  //vec3 n = normalize( gl_NormalMatrix * varyingNormalDirection );
  //vec3 r = reflect( u, n );
  //float m = sqrt( r.x*r.x + r.y*r.y + (r.z+0.1)*(r.z+0.1) );
  //tex_coord.x = (r.x/m + 0.5)/1;
  //tex_coord.y = (r.y/m + 0.5)/1;
}
