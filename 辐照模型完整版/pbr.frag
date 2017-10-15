#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;
in mat3 TBN;

//材质属性
uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

//辐照度
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

//灯光
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = PI * denom * denom;

	return nom / denom;
}

float D_GGX( float Roughness, float NoH )
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return a2 / ( PI*d*d );					// 4 mul, 1 rcp
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;

	float nom = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0f);
	float NdotL = max(dot(N, L), 0.0f);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

float Vis_Schlick( float Roughness, float NoV, float NoL )
{
	float k = Roughness * Roughness * 0.5;
	float Vis_SchlickV = NoV * (1 - k) + k;
	float Vis_SchlickL = NoL * (1 - k) + k;
	return 0.25 / ( Vis_SchlickV * Vis_SchlickL );
}

float Vis_SmithJointApprox( float Roughness, float NoV, float NoL )
{
	float a = Roughness * Roughness;
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	return 2 * ( Vis_SmithV + Vis_SmithL );
}

float Vis_Smith( float Roughness, float NoV, float NoL )
{
	float a = Roughness * Roughness;
	float a2 = a*a;

	float Vis_SmithV = NoV + sqrt( NoV * (NoV - NoV * a2) + a2 );
	float Vis_SmithL = NoL + sqrt( NoL * (NoL - NoL * a2) + a2 );
	return ( Vis_SmithV * Vis_SmithL );
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0f - roughness), F0) - F0) * pow(1.0f - cosTheta, 5.0f);
}   

vec3 F_Schlick(float VoH, vec3 SpecularColor)
{
	float Fc = ( 1 - VoH ) * ( 1 - VoH ) * ( 1 - VoH ) * ( 1 - VoH ) * ( 1 - VoH );					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad
	
	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return clamp(0.0, 1.0, 50.0 * SpecularColor.g ) * Fc + (1 - Fc) * SpecularColor;
}

vec3 F_Fresnel(float VoH, vec3 SpecularColor)
{
	vec3 SpecularColorSqrt = sqrt( clamp( vec3(0, 0, 0), vec3(0.99, 0.99, 0.99), SpecularColor ) );
	vec3 n = ( 1 + SpecularColorSqrt ) / ( 1 - SpecularColorSqrt );
	vec3 g = sqrt( n*n + VoH*VoH - 1 );
	return 0.5 * ((g - VoH) / (g + VoH)) * ((g - VoH) / (g + VoH)) * ( 1 + (((g+VoH)*VoH - 1) / ((g-VoH)*VoH + 1)) * (((g+VoH)*VoH - 1) / ((g-VoH)*VoH + 1)));
}

void main()
{
	vec3 N = Normal;
	vec3 V = normalize(camPos - WorldPos);
	vec3 R = refract(-V, N, 0.75);

	//计算反射比，如果是电解质（比如塑料）让F0为0.04，如果是金属，则使用它们的反射颜色
	vec3 F0 = vec3(0.04f);
	F0 = mix(F0, albedo, metallic);

	//反射方程
	vec3 Lo = vec3(0.0f);
	for (int i = 0; i < 4; i++)
	{
		//计算每个灯光的辐射
		vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

		//双向反射分布函数
		float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);    
        vec3 F    = F_Fresnel(max(dot(H, V), 0.0), F0);        
        
        vec3 nominator    = NDF * G * F;
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
        vec3 brdf = nominator / denominator;

		//折射实验 双向传递分布函数
		float glassno = 1.5;//镜子折射率
		float airni = 1.0;//空气折射率
		float NdotO = abs(dot(N, V));
		float NdotI = abs(dot(N, L));
			
		vec3 ht = -(airni * L + glassno * V);
		vec3 Ht = normalize(ht);
		float HdotO = abs(dot(Ht, V));
		float HdotI = abs(dot(Ht, L));

		float denominatorT = (airni * HdotI + glassno * HdotO);

		vec3 btdf = ((HdotO * HdotI) / (NdotO * NdotI)) *  
					glassno * glassno * (1.0 - F_Schlick(dot(L, Ht), F0)) * 
					Vis_Smith(roughness, HdotO, HdotI) * D_GGX(roughness, dot(N, Ht)) * 
					(1.0 / (denominatorT * denominatorT));

		//brdf += btdf;

		vec3 KS = F;
		vec3 KD = vec3(1.0f) - KS;
		KD *= 1.0f - metallic;

		float NdotL = max(dot(N, L), 0.0f);

		//除以PI是为了标准化
		Lo += (KD * albedo / PI + brdf) * radiance * NdotL;

		//Lo += btdf * radiance * NdotL;
	}

	//环境光（使用辐照度模型）
	vec3 F = F_Fresnel(max(dot(N, V), 0.0), F0);

	vec3 KS = F;
	vec3 KD = 1.0f - KS;
	KD *= 1.0f - metallic;

	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse = irradiance * albedo;

	//我们从预过滤贴图和双向反射分布函数的查找贴图中进行采样，并融和它们的结果作为辐照度的镜面反射部分
	const float MAX_REFLECTION_LOD = 5.0f;
	vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
	vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0f), roughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient = (KD * diffuse + specular) * ao;
	vec3 color = ambient + Lo;

	//HDR色调映射函数
	color = color / (color + vec3(1.0f));

	//伽马校正
	color = pow(color, vec3(1.0f/2.0f));

	FragColor = vec4(color, 0.0f);
}