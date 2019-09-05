varying vec3 varyingImg_coord;
varying vec2 varyingAtlas_coord;
uniform sampler2D tempImg;
uniform sampler2D bufImg;

void main(){
	vec2 img_coord = varyingImg_coord.xy;
	//gl_FragColor = vec4(1.0,1.0,1.0,1.0);
	//gl_FragColor = vec4(texture2D(tempImg, img_coord).rgb, 1.0);
	gl_FragColor =  vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb+varyingImg_coord.z*texture2D(tempImg, img_coord).rgb, 1.0);
	//gl_FragColor = texture2D(bufImg, vec2((gl_PointCoord.x+1.0)/2.0,(-gl_PointCoord.y+1.0)/2.0))+varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy*2-1);
	//gl_FragColor = texture2D(bufImg, vec2((gl_FragCoord.x+1.0)/2.0,(gl_FragCoord.y+1.0)/2.0))+varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy);
	
	//gl_FragColor =vec4(varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy).rgb, 1.0);

}