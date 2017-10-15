#version 330 core
out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;//得到辐照度贴图
uniform float roughness;

const float PI = 3.14159265359f;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)//正态分布函数
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// 有效的 VanDerCorpus 计算.
//VanDerCorpus序列式为1/2，1/4，3/4，1/8，5/8，7/8 
//二进制情况下即为：0.1，0.01，0.11，0.001，0.101
float RadicalInverse_VdC(uint bits) //输入的参数的含义为序列下标
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)//生成Hammersley二维向量
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	//通过粗糙度和Xi向量获得球面坐标
	float phi = 2.0 * PI * Xi.x;        //球面坐标系中的φ即确定水平方向
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));//确定竖直方向
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	
	// 从球面坐标系转变到笛卡尔坐标系 得到 半程向量
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	//将切线空间中的半程向量转变到世界空间中的采样向量
	vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
void main()
{		
    vec3 N = normalize(WorldPos);//法线向量
    
    // 做一个简易的假设，假设V=R=法线 
    vec3 R = N;
    vec3 V = R; //观察向量

    const uint SAMPLE_COUNT = 1024u;
    vec3 prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // 生成一个样本向量，偏向优先对准方向 (重要性采样).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, roughness);//半程向量（重要性采样）
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);   //入射向量

        float NdotL = max(dot(N, L), 0.0);				//入射向量与法线的夹角，即入射光线的强度
        if(NdotL > 0.0)
        {
			//在粗糙度/pdf的基础上从环境贴图的相应mip等级中采样
            float D     = DistributionGGX(N, H, roughness); //获取正态分布函数
            float NdotH = max(dot(N, H), 0.0);				//法线与半程向量点乘获得镜面反射的计算结果
            float HdotV = max(dot(H, V), 0.0);				//
            float pdf   = D * NdotH / (4.0 * HdotV) + 0.0001; //求出pdf函数，即该采样向量的抽取概率值

			
            float resolution = 512.0; // 立方体贴图的原分辨率 (per face)

			//球面度是立体角的国际单位
			//1球面度所对应的立体角所对应的球面表面积为r2。球表面积为4πr2，因此整个球有4π个球面度，即Ω=S/r2。
			//球面度/立方体贴图像素总数目
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);//单位像素对应的球面度
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);//1/该采样向量的可能的数量=？？？

			//                                                单位像素中的采样密度？？？？
            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
			//根据入射光线和mip等级在环境贴图中采样
            prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
			//采样的总权重
            totalWeight      += NdotL;        
        }
    }

    prefilteredColor = prefilteredColor / totalWeight;//求平均值

    FragColor = vec4(prefilteredColor, 1.0);
}