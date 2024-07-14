/** @type {import('next').NextConfig} */
const nextConfig = {
    webpack: (config, { isServer }) => {
        config.externals.push({
            'onnxruntime-node': 'commonjs onnxruntime-node'
        });
        return config;
    },
};

export default nextConfig;
