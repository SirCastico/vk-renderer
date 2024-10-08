//GLSL version to use
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_debug_printf : enable

//size of a workgroup for compute, specialization constants can be used
layout (local_size_x = 8, local_size_y = 8) in;

//descriptor bindings for the pipeline
layout(rgba16f,set = 0, binding = 0) uniform image2D image;

struct AABB{
    vec3 min,max;
};

struct MeshData{
    vec3 color;
    AABB aabb;
    uint inds_start_i, len, vert_offset;
};

struct Vertex{
    vec3 pos;
};

layout(buffer_reference, scalar, buffer_reference_align=8) readonly buffer MeshMetaDataBuffer{
    uint size;
    MeshData meshes[];
};

layout(buffer_reference, scalar, buffer_reference_align=8) readonly buffer VertexBuffer{
    vec3 vertices[];
};

layout(buffer_reference, scalar, buffer_reference_align=8) readonly buffer MeshDataBuffer{
    VertexBuffer verts_ref;
    uint ind_count;
    uint indices[];
};

layout(push_constant, scalar) uniform Data{
    mat4 cam;
    MeshMetaDataBuffer mesh_metadata_ref;
    MeshDataBuffer mesh_data_ref;
    float fov_tan;
}data;

struct Triangle{
    vec3 a,b,c;
};

bool ray_aabb_collision(AABB aabb, vec4 origin, vec4 dir){
    const float tx1 = (aabb.min.x - origin.x) / dir.x;
    const float tx2 = (aabb.max.x - origin.x) / dir.x;

    float tmin = min(tx1,tx2);
    float tmax = max(tx1,tx2);

    float ty1 = (aabb.min.y - origin.y) / dir.y;
    float ty2 = (aabb.max.y - origin.y) / dir.y;

    tmin = max(tmin,min(ty1,ty2));
    tmax = min(tmax,max(ty1,ty2));

    float tz1 = (aabb.min.z - origin.z) / dir.z;
    float tz2 = (aabb.max.z - origin.z) / dir.z;

    tmin = max(tmin,min(tz1,tz2));
    tmax = min(tmax,max(tz1,tz2));

    return tmax>=tmin;
}

const float EPSILON = 0.00001;

bool ray_triangle_intersect(vec3 origin, vec3 dir, Triangle tri, out vec3 isect_pos){
    vec3 edge1 = tri.b - tri.a;
    vec3 edge2 = tri.c - tri.a;
    vec3 ray_cross_e2 = cross(dir, edge2);
    float det = dot(edge1, ray_cross_e2);

    if(det > -EPSILON && det < EPSILON)
        return false;

    float inv_det = 1.0 / det;
    vec3 s = origin - tri.a;
    float u = inv_det * dot(s, ray_cross_e2);

    if (u < 0.0 || u > 1.0)
        return false;

    vec3 s_cross_e1 = cross(s, edge1);
    float v = inv_det * dot(dir, s_cross_e1);

    if (v < 0.0 || u + v > 1.0)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * dot(edge2, s_cross_e1);

    if (t > EPSILON) // ray intersection
    {
        isect_pos = vec3(origin + dir * t);
        return true;
    }else // This means that there is a line intersection but not a ray intersection.
        return false;
}


#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308

float length_squared_vec3(vec3 v){
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

vec4 render(vec4 origin, vec4 dir){
    vec4 color = vec4(0.0,0.0,0.0,1.0);
    float current_len_sq = FLT_MAX;
    int current_mesh = -1;

    for(uint mesh_i=0;mesh_i<data.mesh_metadata_ref.size;++mesh_i){
        MeshData mesh_data = data.mesh_metadata_ref.meshes[mesh_i];
        debugPrintfEXT("%f,%f,%f", mesh_data.color.r, mesh_data.color.g, mesh_data.color.b);

        if(ray_aabb_collision(mesh_data.aabb, origin, dir)){
            for(uint v_ind=mesh_data.inds_start_i;v_ind<mesh_data.inds_start_i+mesh_data.len;v_ind+=3){
                uint ind0 = data.mesh_data_ref.indices[v_ind];
                uint ind1 = data.mesh_data_ref.indices[v_ind+1];
                uint ind2 = data.mesh_data_ref.indices[v_ind+2];

                Triangle tri = {
                    data.mesh_data_ref.verts_ref.vertices[ind0 + mesh_data.vert_offset],
                    data.mesh_data_ref.verts_ref.vertices[ind1 + mesh_data.vert_offset],
                    data.mesh_data_ref.verts_ref.vertices[ind2 + mesh_data.vert_offset],
                };

                vec3 isect_pos;
                if(ray_triangle_intersect(origin.xyz, dir.xyz, tri, isect_pos)){
                    vec3 origin_isect_vec = isect_pos - origin.xyz;
                    float len = length_squared_vec3(origin_isect_vec);
                    if(len < current_len_sq) {
                        current_len_sq = len;
                        current_mesh = int(mesh_i);
                    }
                }
            }
        }
    }

    if(current_mesh>=0)
        color = vec4(data.mesh_metadata_ref.meshes[current_mesh].color,1.0);

    return color;
}

void main() {
    ivec2 image_size = imageSize(image);
    if(gl_GlobalInvocationID.x>=image_size.x || gl_GlobalInvocationID.y>=image_size.y)
        return;

    vec2 texel_coord = vec2(gl_GlobalInvocationID.xy);
    float aspect_ratio = float(image_size.x)/float(image_size.y);

    float px = (2.0*((texel_coord.x+0.5)/image_size.x)-1.0) * data.fov_tan * aspect_ratio;
    float py = (1.0-2.0*((texel_coord.y+0.5)/image_size.y)) * data.fov_tan;

    vec4 ray_origin =  data.cam * vec4(0.0,0.0,0.0,1.0);
    vec4 ray_dir =  data.cam * normalize(vec4(px,py,-1.0,0.0));
    ray_dir = normalize(ray_dir);
    
    vec4 color = render(ray_origin,ray_dir);

    imageStore(image, ivec2(gl_GlobalInvocationID.xy), color);
}
