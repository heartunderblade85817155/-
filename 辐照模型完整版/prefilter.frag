#version 330 core
out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;//�õ����ն���ͼ
uniform float roughness;

const float PI = 3.14159265359f;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)//��̬�ֲ�����
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
// ��Ч�� VanDerCorpus ����.
//VanDerCorpus����ʽΪ1/2��1/4��3/4��1/8��5/8��7/8 
//����������¼�Ϊ��0.1��0.01��0.11��0.001��0.101
float RadicalInverse_VdC(uint bits) //����Ĳ����ĺ���Ϊ�����±�
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)//����Hammersley��ά����
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	//ͨ���ֲڶȺ�Xi���������������
	float phi = 2.0 * PI * Xi.x;        //��������ϵ�еĦռ�ȷ��ˮƽ����
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));//ȷ����ֱ����
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	
	// ����������ϵת�䵽�ѿ�������ϵ �õ� �������
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	//�����߿ռ��еİ������ת�䵽����ռ��еĲ�������
	vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
void main()
{		
    vec3 N = normalize(WorldPos);//��������
    
    // ��һ�����׵ļ��裬����V=R=���� 
    vec3 R = N;
    vec3 V = R; //�۲�����

    const uint SAMPLE_COUNT = 1024u;
    vec3 prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // ����һ������������ƫ�����ȶ�׼���� (��Ҫ�Բ���).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, roughness);//�����������Ҫ�Բ�����
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);   //��������

        float NdotL = max(dot(N, L), 0.0);				//���������뷨�ߵļнǣ���������ߵ�ǿ��
        if(NdotL > 0.0)
        {
			//�ڴֲڶ�/pdf�Ļ����ϴӻ�����ͼ����Ӧmip�ȼ��в���
            float D     = DistributionGGX(N, H, roughness); //��ȡ��̬�ֲ�����
            float NdotH = max(dot(N, H), 0.0);				//��������������˻�þ��淴��ļ�����
            float HdotV = max(dot(H, V), 0.0);				//
            float pdf   = D * NdotH / (4.0 * HdotV) + 0.0001; //���pdf���������ò��������ĳ�ȡ����ֵ

			
            float resolution = 512.0; // ��������ͼ��ԭ�ֱ��� (per face)

			//�����������ǵĹ��ʵ�λ
			//1���������Ӧ�����������Ӧ����������Ϊr2��������Ϊ4��r2�������������4�и�����ȣ�����=S/r2��
			//�����/��������ͼ��������Ŀ
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);//��λ���ض�Ӧ�������
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);//1/�ò��������Ŀ��ܵ�����=������

			//                                                ��λ�����еĲ����ܶȣ�������
            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
			//����������ߺ�mip�ȼ��ڻ�����ͼ�в���
            prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
			//��������Ȩ��
            totalWeight      += NdotL;        
        }
    }

    prefilteredColor = prefilteredColor / totalWeight;//��ƽ��ֵ

    FragColor = vec4(prefilteredColor, 1.0);
}