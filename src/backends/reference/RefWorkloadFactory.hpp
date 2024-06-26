//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RefMemoryManager.hpp"

#include <armnn/Optional.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/utility/IgnoreUnused.hpp>


namespace armnn
{

template <typename QueueDescriptorType>
constexpr bool IsOperationQueueDescriptor(const QueueDescriptorType&) { return true; }
template <>
constexpr bool IsOperationQueueDescriptor(const MemCopyQueueDescriptor&) { return false; }
template <>
constexpr bool IsOperationQueueDescriptor(const ConstantQueueDescriptor&) { return false; }
template <>
constexpr bool IsOperationQueueDescriptor(const PermuteQueueDescriptor&) { return false; }

// Reference workload factory.
class RefWorkloadFactory : public IWorkloadFactory
{
public:
    explicit RefWorkloadFactory(const std::shared_ptr<RefMemoryManager>& memoryManager);
    RefWorkloadFactory();

    ~RefWorkloadFactory() {}

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported,
                                 const ModelOptions& modelOptions);

    bool SupportsSubTensors() const override { return false; }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override
    {
        IgnoreUnused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<IWorkload> CreateWorkload(LayerType type,
                                              const QueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

private:
    template <typename F32Workload, typename U8Workload, typename QueueDescriptorType>
    std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor, const WorkloadInfo& info) const;

    mutable std::shared_ptr<RefMemoryManager> m_MemoryManager;
};

} // namespace armnn
