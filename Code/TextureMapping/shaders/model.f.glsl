/**
 * From the OpenGL Programming wikibook: http://en.wikibooks.org/wiki/OpenGL_Programming
 * This file is in the public domain.
 * Contributors: Martin Kraus, Sylvain Beucler
 */
varying vec4 position;  // position of the vertex (and fragment) in world space
varying vec3 varyingNormalDirection;  // surface normal vector in world space
varying vec2 tex_coord;
varying vec3 img_tex_coord;
uniform sampler2D tex;
uniform sampler2D tex_s;
uniform sampler2D tex_n;
uniform sampler2D imgTex;
uniform mat4 m, v, p;
uniform vec4 flags;
uniform mat4 v_inv;

struct lightSource
{
  vec4 position;
  vec4 diffuse;
  vec4 specular;
  float constantAttenuation, linearAttenuation, quadraticAttenuation;
  float spotCutoff, spotExponent;
  vec3 spotDirection;
};
lightSource light0 = lightSource(
  vec4(0.0, 0.0, -2.0, 1.0),
  vec4(1.0,  1.0,  1.0, 1.0),
  vec4(1.0,  1.0,  1.0, 1.0),
  0.0, 0.5, 0.0,
  180.0, 0.0,
  vec3(0.0, 0.0, 0.0)
);
vec4 scene_ambient = vec4(0.2, 0.2, 0.2, 1.0);

struct material
{
  vec4 ambient;
  vec4 diffuse;
  vec4 specular;
  float shininess;
};
material frontMaterial = material(
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  15.0
);
material backMaterial = material(
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  vec4(0.50754f, 0.50754f, 0.50754f, 1.0),
  15.0
);

void main()
{
  vec3 normalDirection = normalize(varyingNormalDirection);
  vec4 ambientColor;
  vec4 diffuseColor;
  vec4 specularColor;
  float shininess;
  
  if (gl_FrontFacing)
    {
      ambientColor = frontMaterial.ambient;
      diffuseColor = frontMaterial.diffuse;
      specularColor = frontMaterial.specular;
      shininess = frontMaterial.shininess;
    }
  else
    {
      ambientColor = backMaterial.ambient;
      diffuseColor = backMaterial.diffuse;
      specularColor = backMaterial.specular;
      shininess = backMaterial.shininess;
      normalDirection = -normalDirection;
    }

  
  vec3 viewDirection = normalize(vec3(v_inv * vec4(0.0, 0.0, 0.0, 1.0) - position));
  vec3 lightDirection;
  float attenuation;
  
  if (0.0 == light0.position.w) // directional light?
    {
      attenuation = 1.0; // no attenuation
      lightDirection = normalize(vec3(light0.position));
    } 
  else // point light or spotlight (or other kind of light) 
    {
      vec3 positionToLightSource = vec3(light0.position - position);
      float distance = length(positionToLightSource);
      lightDirection = normalize(positionToLightSource);
      attenuation = 1.0 / (light0.constantAttenuation
			   + light0.linearAttenuation * distance
			   + light0.quadraticAttenuation * distance * distance);
      
      if (light0.spotCutoff <= 90.0) // spotlight?
	{
	  float clampedCosine = max(0.0, dot(-lightDirection, light0.spotDirection));
	  if (clampedCosine < cos(radians(light0.spotCutoff))) // outside of spotlight cone?
	    {
	      attenuation = 0.0;
	    }
	  else
	    {
	      attenuation = attenuation * pow(clampedCosine, light0.spotExponent); 
	    }
	}
    }
  
  vec3 ambientLighting = vec3(scene_ambient) * vec3(ambientColor);
  
  vec3 diffuseReflection = attenuation 
    * vec3(light0.diffuse) * vec3(diffuseColor)
    * max(0.0, dot(normalDirection, lightDirection));
 
  vec3 specularReflection;
  if (dot(normalDirection, lightDirection) < 0.0) // light source on the wrong side?
    {
      specularReflection = vec3(0.0, 0.0, 0.0); // no specular reflection
    }
  else // light source on the right side
    {
      specularReflection = attenuation * vec3(light0.specular) * vec3(specularColor) 
	* pow(max(0.0, dot(reflect(-lightDirection, normalDirection), viewDirection)), shininess);
    }
 // if(tex_coord.z == 0)
 // {
	//gl_FragColor = 0.5*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.5*vec4(0.0,0.0,1.0,1.0);
 // }
  vec4 resultColor;
  if(flags[1] == 0){
	if(flags[0] == 0)
	  {
		resultColor = vec4(ambientLighting + diffuseReflection + specularReflection, 1.0);//0.01*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.99*texture2D(tex, tex_coord.xy);
	}
	else{
		resultColor = 0.01*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.99*texture2D(tex_n, tex_coord.xy);
	}
	//resultColor = 0.1*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.9*texture2D(tex, tex_coord.xy);
  }
  else
  {
	if(flags[0] == 0)
	  {
		//resultColor = 0.01*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.99*((1.0-img_tex_coord.z)*texture2D(tex, tex_coord.xy) + img_tex_coord.z*texture2D(imgTex, img_tex_coord.xy));
		resultColor = 0.01*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.99*texture2D(tex, tex_coord.xy);
	  }
	  else
	  {
		resultColor = 0.01*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.99*texture2D(tex_s, tex_coord.xy);
		//resultColor = 0.009*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.001*((1.0-img_tex_coord.z)*texture2D(tex, tex_coord.xy) + img_tex_coord.z*texture2D(imgTex, img_tex_coord.xy)) + 0.99*((1.0-img_tex_coord.z)*vec4(1.0,0.0,0.0, 1.0) + img_tex_coord.z*vec4(0.0,0.0,1.0, 1.0));
	  }
  }
  

  gl_FragColor = resultColor;
  //gl_FragColor = 0.09*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.01*((1-img_tex_coord.z)*texture2D(tex, tex_coord.xy) + img_tex_coord.z*texture2D(imgTex, img_tex_coord.xy)) + 0.9*((1-img_tex_coord.z)*vec4(1.0,0.0,0.0, 1.0) + img_tex_coord.z*vec4(0.0,0.0,1.0, 1.0));
  //gl_FragColor = 0.1*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.9*((1-img_tex_coord.z)*texture2D(tex, tex_coord.xy) + img_tex_coord.z*texture2D(imgTex, img_tex_coord.xy));
  //gl_FragColor = 0.1*vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + 0.9*texture2D(tex, tex_coord.xy) + 0.001*img_tex_coord.z*texture2D(imgTex, img_tex_coord.xy);
  //gl_FragColor = vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) * texture2D(tex, tex_coord);
  //gl_FragColor = vec4(ambientLighting + diffuseReflection + specularReflection, 1.0);
  //gl_FragColor = 0.5*vec4(0.0,0.0,0.0,1.0)+0.5*texture2D(tex, tex_coord);
}
