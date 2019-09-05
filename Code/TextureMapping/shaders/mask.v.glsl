attribute vec2 atlas_coord;
attribute vec3 img_coord;
varying vec3 varyingImg_coord;
varying vec2 varyingAtlas_coord;
uniform vec2 img_size;

void main()
{
	gl_Position = vec4(atlas_coord.x*2.0-1.0, -(atlas_coord.y*2.0-1.0), 0.0, 1.0);
	//gl_Position = vec4(atlas_coord.x, (atlas_coord.y), 0.0, 1.0);
	//varyingImg_coord = vec3(((img_coord.x/2048.0)*2.0)-1.0, ((img_coord.y/1536.0)*2.0-1.0), img_coord.z);
	//varyingImg_coord = vec3((img_coord.x/2048.0), ((img_coord.y/1536.0)), img_coord.z);
	//varyingImg_coord = vec3((img_coord.x/640.0), ((img_coord.y/480.0)), img_coord.z);
	varyingImg_coord = vec3((img_coord.x/img_size.x), ((img_coord.y/img_size.y)), img_coord.z);
	varyingAtlas_coord = atlas_coord;
	//varyingImg_coord = vec3((img_coord.x)*2.0-1.0, (img_coord.y)*2.0-1.0, img_coord.z);
}